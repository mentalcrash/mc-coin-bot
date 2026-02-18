"""OMS 테스트.

validated 주문 처리, 멱등성, CircuitBreaker 전량 청산을 검증합니다.
"""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from src.core.event_bus import EventBus
from src.core.events import (
    AnyEvent,
    BarEvent,
    CircuitBreakerEvent,
    EventType,
    FillEvent,
    OrderAckEvent,
    OrderRejectedEvent,
    OrderRequestEvent,
)
from src.eda.executors import BacktestExecutor, ShadowExecutor
from src.eda.oms import OMS
from src.eda.portfolio_manager import EDAPortfolioManager
from src.portfolio.config import PortfolioManagerConfig
from src.portfolio.cost_model import CostModel


def _make_validated_order(
    symbol: str = "BTC/USDT",
    side: str = "BUY",
    notional_usd: float = 5000.0,
    client_order_id: str | None = None,
) -> OrderRequestEvent:
    return OrderRequestEvent(
        client_order_id=client_order_id or f"test-{uuid4().hex[:8]}",
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        target_weight=0.5,
        notional_usd=notional_usd,
        validated=True,
        correlation_id=uuid4(),
        source="test",
    )


class TestOMSBasics:
    """OMS 기본 동작 테스트."""

    async def test_validated_order_produces_fill(self) -> None:
        """validated 주문(price 설정) → Executor 즉시 실행 → FillEvent 발행."""
        cost_model = CostModel.zero()
        executor = BacktestExecutor(cost_model=cost_model)
        # 가격 데이터 설정
        from src.core.events import BarEvent

        executor.on_bar(
            BarEvent(
                symbol="BTC/USDT",
                timeframe="1D",
                open=50000.0,
                high=51000.0,
                low=49000.0,
                close=50500.0,
                volume=1000.0,
                bar_timestamp=datetime.now(UTC),
            )
        )

        oms = OMS(executor=executor)
        bus = EventBus(queue_size=100)
        fills: list[FillEvent] = []
        acks: list[OrderAckEvent] = []

        async def fill_handler(event: AnyEvent) -> None:
            if isinstance(event, FillEvent):
                fills.append(event)

        async def ack_handler(event: AnyEvent) -> None:
            if isinstance(event, OrderAckEvent):
                acks.append(event)

        bus.subscribe(EventType.FILL, fill_handler)
        bus.subscribe(EventType.ORDER_ACK, ack_handler)
        await oms.register(bus)

        task = asyncio.create_task(bus.start())
        # SL/TS 스타일 주문 (price 설정 → 즉시 체결)
        order = OrderRequestEvent(
            client_order_id="test-sl-1",
            symbol="BTC/USDT",
            side="BUY",
            target_weight=0.5,
            notional_usd=5000.0,
            price=50000.0,
            validated=True,
            correlation_id=uuid4(),
            source="test",
        )
        await bus.publish(order)
        await bus.stop()
        await task

        assert len(acks) == 1
        assert len(fills) == 1
        assert fills[0].fill_price == 50000.0
        assert oms.total_fills == 1

    async def test_deferred_order_no_immediate_fill(self) -> None:
        """일반 주문(price=None) → deferred, 즉시 FillEvent 없음."""
        cost_model = CostModel.zero()
        executor = BacktestExecutor(cost_model=cost_model)
        from src.core.events import BarEvent

        executor.on_bar(
            BarEvent(
                symbol="BTC/USDT",
                timeframe="1D",
                open=50000.0,
                high=51000.0,
                low=49000.0,
                close=50500.0,
                volume=1000.0,
                bar_timestamp=datetime.now(UTC),
            )
        )

        oms = OMS(executor=executor)
        bus = EventBus(queue_size=100)
        fills: list[FillEvent] = []

        async def fill_handler(event: AnyEvent) -> None:
            if isinstance(event, FillEvent):
                fills.append(event)

        bus.subscribe(EventType.FILL, fill_handler)
        await oms.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_validated_order())
        await bus.stop()
        await task

        assert len(fills) == 0  # deferred, no immediate fill
        assert executor.pending_count == 1  # order stored as pending

    async def test_unvalidated_order_ignored(self) -> None:
        """validated=False인 주문은 무시."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        oms = OMS(executor=executor)
        bus = EventBus(queue_size=100)
        fills: list[FillEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, FillEvent):
                fills.append(event)

        bus.subscribe(EventType.FILL, handler)
        await oms.register(bus)

        task = asyncio.create_task(bus.start())
        # validated=False 주문
        unvalidated = OrderRequestEvent(
            client_order_id="test-1",
            symbol="BTC/USDT",
            side="BUY",
            target_weight=0.5,
            notional_usd=5000.0,
            validated=False,
            correlation_id=uuid4(),
            source="test",
        )
        await bus.publish(unvalidated)
        await bus.stop()
        await task

        assert len(fills) == 0

    async def test_idempotency(self) -> None:
        """동일 client_order_id 중복 주문 무시."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        from src.core.events import BarEvent

        executor.on_bar(
            BarEvent(
                symbol="BTC/USDT",
                timeframe="1D",
                open=50000.0,
                high=51000.0,
                low=49000.0,
                close=50500.0,
                volume=1000.0,
                bar_timestamp=datetime.now(UTC),
            )
        )

        oms = OMS(executor=executor)
        bus = EventBus(queue_size=100)
        fills: list[FillEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, FillEvent):
                fills.append(event)

        bus.subscribe(EventType.FILL, handler)
        await oms.register(bus)

        task = asyncio.create_task(bus.start())
        # 같은 ID로 SL 스타일 주문 2번 발행
        order = OrderRequestEvent(
            client_order_id="dup-order-1",
            symbol="BTC/USDT",
            side="BUY",
            target_weight=0.5,
            notional_usd=5000.0,
            price=50000.0,
            validated=True,
            correlation_id=uuid4(),
            source="test",
        )
        await bus.publish(order)
        await bus.publish(order)
        await bus.stop()
        await task

        assert len(fills) == 1  # 1번만 체결
        assert oms.total_fills == 1
        assert oms.total_rejected == 1  # 중복 1건 거부


class TestShadowExecutor:
    """ShadowExecutor 테스트."""

    async def test_shadow_logs_no_fill(self) -> None:
        """ShadowExecutor는 로깅만, FillEvent 없음."""
        executor = ShadowExecutor()
        oms = OMS(executor=executor)
        bus = EventBus(queue_size=100)
        fills: list[FillEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, FillEvent):
                fills.append(event)

        bus.subscribe(EventType.FILL, handler)
        await oms.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_validated_order())
        await bus.stop()
        await task

        assert len(fills) == 0
        assert len(executor.order_log) == 1


class TestCircuitBreakerClose:
    """CircuitBreaker 전량 청산 테스트."""

    async def test_circuit_breaker_closes_positions(self) -> None:
        """CB 이벤트 → 오픈 포지션 전량 청산."""
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=100000.0)
        cost_model = CostModel.zero()
        executor = BacktestExecutor(cost_model=cost_model)

        from src.core.events import BarEvent

        executor.on_bar(
            BarEvent(
                symbol="BTC/USDT",
                timeframe="1D",
                open=50000.0,
                high=51000.0,
                low=49000.0,
                close=50500.0,
                volume=1000.0,
                bar_timestamp=datetime.now(UTC),
            )
        )

        oms = OMS(executor=executor, portfolio_manager=pm)
        bus = EventBus(queue_size=100)
        fills: list[FillEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, FillEvent):
                fills.append(event)

        bus.subscribe(EventType.FILL, handler)
        await pm.register(bus)
        await oms.register(bus)

        task = asyncio.create_task(bus.start())
        # 포지션 생성 (PM fill 통해)
        buy_fill = FillEvent(
            client_order_id="init-1",
            symbol="BTC/USDT",
            side="BUY",
            fill_price=50000.0,
            fill_qty=0.5,
            fee=0.0,
            fill_timestamp=datetime.now(UTC),
            correlation_id=uuid4(),
            source="test",
        )
        await bus.publish(buy_fill)

        # CircuitBreaker 발행
        cb = CircuitBreakerEvent(
            reason="test stop-loss",
            close_all_positions=True,
            correlation_id=uuid4(),
            source="test",
        )
        await bus.publish(cb)
        await bus.stop()
        await task

        # PM이 처리한 초기 fill + CB 청산 fill
        assert len(fills) >= 2
        # 마지막 fill은 SELL (청산)
        close_fill = fills[-1]
        assert close_fill.side == "SELL"


class TestOMSPersistence:
    """OMS 상태 저장/복구 테스트."""

    def test_processed_orders_property(self) -> None:
        """processed_orders 프로퍼티가 내부 set을 반환."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        oms = OMS(executor)
        assert isinstance(oms.processed_orders, set)
        assert len(oms.processed_orders) == 0

    def test_restore_processed_orders(self) -> None:
        """restore_processed_orders로 주문 ID 복원."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        oms = OMS(executor)

        saved_ids = {"order-1", "order-2", "order-3"}
        oms.restore_processed_orders(saved_ids)

        assert oms.processed_orders == saved_ids

    async def test_restored_orders_prevent_duplicates(self) -> None:
        """복원된 주문 ID는 중복 실행 방지."""
        cost_model = CostModel.zero()
        executor = BacktestExecutor(cost_model=cost_model)
        executor.on_bar(
            BarEvent(
                symbol="BTC/USDT",
                timeframe="1D",
                open=50000.0,
                high=51000.0,
                low=49000.0,
                close=50500.0,
                volume=1000.0,
                bar_timestamp=datetime.now(UTC),
                correlation_id=uuid4(),
                source="test",
            )
        )

        oms = OMS(executor)

        # 이미 처리된 주문 ID 복원
        oms.restore_processed_orders({"dup-order-1"})

        bus = EventBus(queue_size=100)
        fills: list[FillEvent] = []

        async def fill_handler(event: AnyEvent) -> None:
            if isinstance(event, FillEvent):
                fills.append(event)

        bus.subscribe(EventType.FILL, fill_handler)
        await oms.register(bus)

        task = asyncio.create_task(bus.start())

        # 복원된 ID로 주문 → 중복 차단
        dup_order = OrderRequestEvent(
            client_order_id="dup-order-1",
            symbol="BTC/USDT",
            side="BUY",
            target_weight=0.5,
            notional_usd=5000.0,
            price=50000.0,
            validated=True,
            correlation_id=uuid4(),
            source="test",
        )
        await bus.publish(dup_order)
        await bus.flush()

        await bus.stop()
        await task

        assert len(fills) == 0  # 중복 주문 → 실행 안 됨
        assert oms.total_rejected == 1


class TestOMSDuplicateRejectionEvent:
    """OMS 중복 주문 시 OrderRejectedEvent 발행 테스트."""

    async def test_duplicate_publishes_rejected_event(self) -> None:
        """동일 client_order_id 중복 주문 시 OrderRejectedEvent 발행."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(
            BarEvent(
                symbol="BTC/USDT",
                timeframe="1D",
                open=50000.0,
                high=51000.0,
                low=49000.0,
                close=50500.0,
                volume=1000.0,
                bar_timestamp=datetime.now(UTC),
            )
        )

        oms = OMS(executor=executor)
        bus = EventBus(queue_size=100)
        rejections: list[OrderRejectedEvent] = []

        async def rejection_handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRejectedEvent):
                rejections.append(event)

        bus.subscribe(EventType.ORDER_REJECTED, rejection_handler)
        await oms.register(bus)

        task = asyncio.create_task(bus.start())

        # SL 스타일 주문 (price 설정 → 즉시 체결)
        order = OrderRequestEvent(
            client_order_id="dup-rej-1",
            symbol="BTC/USDT",
            side="BUY",
            target_weight=0.5,
            notional_usd=5000.0,
            price=50000.0,
            validated=True,
            correlation_id=uuid4(),
            source="test",
        )
        await bus.publish(order)
        await bus.flush()
        # 동일 ID로 재발행
        await bus.publish(order)
        await bus.flush()

        await bus.stop()
        await task

        assert len(rejections) == 1
        assert rejections[0].client_order_id == "dup-rej-1"
        assert "duplicate" in rejections[0].reason.lower()
        assert oms.total_rejected == 1
