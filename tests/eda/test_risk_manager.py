"""EDA RiskManager 테스트.

Pre-trade 검증, 서킷 브레이커 트리거, 무한루프 방지를 검증합니다.
"""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from src.core.event_bus import EventBus
from src.core.events import (
    AnyEvent,
    BalanceUpdateEvent,
    CircuitBreakerEvent,
    EventType,
    FillEvent,
    OrderRejectedEvent,
    OrderRequestEvent,
)
from src.eda.portfolio_manager import EDAPortfolioManager
from src.eda.risk_manager import EDARiskManager
from src.portfolio.config import PortfolioManagerConfig


def _make_order(
    symbol: str = "BTC/USDT",
    side: str = "BUY",
    target_weight: float = 0.5,
    notional_usd: float = 5000.0,
    validated: bool = False,
) -> OrderRequestEvent:
    return OrderRequestEvent(
        client_order_id=f"test-{uuid4().hex[:8]}",
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        target_weight=target_weight,
        notional_usd=notional_usd,
        validated=validated,
        correlation_id=uuid4(),
        source="test",
    )


def _make_balance(
    total_equity: float = 10000.0,
    available_cash: float = 5000.0,
) -> BalanceUpdateEvent:
    return BalanceUpdateEvent(
        total_equity=total_equity,
        available_cash=available_cash,
        total_margin_used=total_equity - available_cash,
        correlation_id=uuid4(),
        source="test",
    )


def _make_fill(
    symbol: str = "BTC/USDT",
    side: str = "BUY",
    price: float = 50000.0,
    qty: float = 0.1,
    fee: float = 0.0,
) -> FillEvent:
    return FillEvent(
        client_order_id="test-fill-1",
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        fill_price=price,
        fill_qty=qty,
        fee=fee,
        fill_timestamp=datetime.now(UTC),
        correlation_id=uuid4(),
        source="test",
    )


class TestOrderValidation:
    """주문 사전 검증 테스트."""

    async def test_valid_order_passes(self) -> None:
        """정상 주문은 validated=True로 재발행."""
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        rm = EDARiskManager(config=config, portfolio_manager=pm)
        bus = EventBus(queue_size=100)
        validated_orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent) and event.validated:
                validated_orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)
        await rm.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_order(notional_usd=5000.0))
        await bus.stop()
        await task

        assert len(validated_orders) == 1
        assert validated_orders[0].validated is True

    async def test_already_validated_ignored(self) -> None:
        """validated=True인 주문은 RM이 무시 (무한루프 방지)."""
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        rm = EDARiskManager(config=config, portfolio_manager=pm)
        bus = EventBus(queue_size=100)
        rejects: list[OrderRejectedEvent] = []
        validated_orders: list[OrderRequestEvent] = []

        async def reject_handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRejectedEvent):
                rejects.append(event)

        async def order_handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent) and event.validated:
                validated_orders.append(event)

        bus.subscribe(EventType.ORDER_REJECTED, reject_handler)
        bus.subscribe(EventType.ORDER_REQUEST, order_handler)
        await pm.register(bus)
        await rm.register(bus)

        task = asyncio.create_task(bus.start())
        # validated=True 주문 발행 → RM이 무시해야 함
        await bus.publish(_make_order(validated=True))
        await bus.stop()
        await task

        assert len(rejects) == 0
        # validated 주문은 그대로 전달 (RM이 재발행하지 않음)
        assert len(validated_orders) == 1

    async def test_max_order_size_rejection(self) -> None:
        """단일 주문 크기 초과 시 거부."""
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        rm = EDARiskManager(
            config=config,
            portfolio_manager=pm,
            max_order_size_usd=1000.0,
        )
        bus = EventBus(queue_size=100)
        rejects: list[OrderRejectedEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRejectedEvent):
                rejects.append(event)

        bus.subscribe(EventType.ORDER_REJECTED, handler)
        await pm.register(bus)
        await rm.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_order(notional_usd=5000.0))
        await bus.stop()
        await task

        assert len(rejects) == 1
        assert "Order size" in rejects[0].reason

    async def test_max_open_positions_rejection(self) -> None:
        """최대 포지션 수 초과 시 거부."""
        config = PortfolioManagerConfig(max_leverage_cap=5.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=100000.0)
        rm = EDARiskManager(
            config=config,
            portfolio_manager=pm,
            max_open_positions=2,
        )
        bus = EventBus(queue_size=100)
        rejects: list[OrderRejectedEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRejectedEvent):
                rejects.append(event)

        bus.subscribe(EventType.ORDER_REJECTED, handler)
        await pm.register(bus)
        await rm.register(bus)

        task = asyncio.create_task(bus.start())
        # 2개 포지션 생성 (PM fill을 통해)
        await bus.publish(_make_fill(symbol="BTC/USDT", price=50000.0, qty=0.1))
        await bus.publish(_make_fill(symbol="ETH/USDT", price=3000.0, qty=1.0))
        # 3번째 심볼 주문 → 거부
        await bus.publish(_make_order(symbol="SOL/USDT", notional_usd=500.0))
        await bus.stop()
        await task

        assert len(rejects) == 1
        assert "Max open positions" in rejects[0].reason


class TestCircuitBreaker:
    """서킷 브레이커 테스트."""

    async def test_stop_loss_triggers_circuit_breaker(self) -> None:
        """Drawdown이 system_stop_loss 초과 시 서킷 브레이커 발동."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=0.10,
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        rm = EDARiskManager(config=config, portfolio_manager=pm)
        bus = EventBus(queue_size=100)
        cb_events: list[CircuitBreakerEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, CircuitBreakerEvent):
                cb_events.append(event)

        bus.subscribe(EventType.CIRCUIT_BREAKER, handler)
        await pm.register(bus)
        await rm.register(bus)

        task = asyncio.create_task(bus.start())
        # Peak equity = 10000, equity 8500 → drawdown 15% > 10%
        await bus.publish(_make_balance(total_equity=8500.0, available_cash=5000.0))
        await bus.stop()
        await task

        assert len(cb_events) == 1
        assert rm.is_circuit_breaker_active is True
        assert "stop-loss" in cb_events[0].reason

    async def test_circuit_breaker_rejects_all_orders(self) -> None:
        """서킷 브레이커 발동 후 모든 주문 거부."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=0.10,
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        rm = EDARiskManager(config=config, portfolio_manager=pm)
        bus = EventBus(queue_size=100)
        rejects: list[OrderRejectedEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRejectedEvent):
                rejects.append(event)

        bus.subscribe(EventType.ORDER_REJECTED, handler)
        await pm.register(bus)
        await rm.register(bus)

        task = asyncio.create_task(bus.start())
        # 서킷 브레이커 발동
        await bus.publish(_make_balance(total_equity=8000.0))
        # 주문 시도 → 거부
        await bus.publish(_make_order(notional_usd=1000.0))
        await bus.stop()
        await task

        assert len(rejects) == 1
        assert "Circuit breaker" in rejects[0].reason

    async def test_no_stop_loss_no_circuit_breaker(self) -> None:
        """system_stop_loss=None이면 서킷 브레이커 비활성화."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=None,
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        rm = EDARiskManager(config=config, portfolio_manager=pm)
        bus = EventBus(queue_size=100)
        cb_events: list[CircuitBreakerEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, CircuitBreakerEvent):
                cb_events.append(event)

        bus.subscribe(EventType.CIRCUIT_BREAKER, handler)
        await pm.register(bus)
        await rm.register(bus)

        task = asyncio.create_task(bus.start())
        # 큰 drawdown이어도 발동 안 함
        await bus.publish(_make_balance(total_equity=5000.0))
        await bus.stop()
        await task

        assert len(cb_events) == 0
        assert rm.is_circuit_breaker_active is False


class TestPeakEquityTracking:
    """Peak equity 추적 테스트."""

    async def test_peak_equity_updates_on_new_high(self) -> None:
        """Equity 고점 갱신 시 peak 업데이트."""
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        rm = EDARiskManager(config=config, portfolio_manager=pm)
        bus = EventBus(queue_size=100)
        await pm.register(bus)
        await rm.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_balance(total_equity=12000.0))
        await bus.publish(_make_balance(total_equity=11000.0))
        await bus.publish(_make_balance(total_equity=13000.0))
        await bus.stop()
        await task

        assert rm.peak_equity == 13000.0

    def test_initial_peak_equity(self) -> None:
        """초기 peak equity = PM total_equity."""
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=25000.0)
        rm = EDARiskManager(config=config, portfolio_manager=pm)
        assert rm.peak_equity == 25000.0

    def test_drawdown_calculation(self) -> None:
        """Drawdown 계산 정확성."""
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        rm = EDARiskManager(config=config, portfolio_manager=pm)
        # Peak = 10000, current equity = 10000 → drawdown = 0%
        assert rm.current_drawdown == 0.0
