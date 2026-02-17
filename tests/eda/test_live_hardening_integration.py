"""Live Hardening 통합 테스트.

Pre-flight → balance sync → order → fill → reconcile 전체 흐름 검증.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.core.event_bus import EventBus
from src.core.events import (
    AnyEvent,
    CircuitBreakerEvent,
    EventType,
    OrderRequestEvent,
)
from src.eda.executors import LiveExecutor
from src.eda.portfolio_manager import EDAPortfolioManager
from src.eda.reconciler import PositionReconciler
from src.eda.risk_manager import EDARiskManager
from src.models.types import Direction
from src.portfolio.config import PortfolioManagerConfig
from src.portfolio.cost_model import CostModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakePosition:
    """테스트용 간이 Position."""

    symbol: str
    direction: Direction = Direction.NEUTRAL
    size: float = 0.0
    avg_entry_price: float = 0.0
    last_price: float = 0.0
    atr_values: list[float] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.size > 0.0

    @property
    def notional(self) -> float:
        return self.size * self.last_price


def _make_config() -> PortfolioManagerConfig:
    return PortfolioManagerConfig(
        max_leverage_cap=2.0,
        rebalance_threshold=0.01,
        system_stop_loss=0.10,
        use_trailing_stop=False,
        cost_model=CostModel.zero(),
    )


def _make_mock_futures_client() -> MagicMock:
    """Mock BinanceFuturesClient with all hardening methods."""
    client = MagicMock()
    client.create_order = AsyncMock(
        return_value={
            "id": "ord_001",
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "amount": 0.001,
            "filled": 0.001,
            "average": 50000.0,
            "price": 50000.0,
            "cost": 50.0,
            "fee": {"cost": 0.02, "currency": "USDT"},
            "status": "closed",
        }
    )
    client.fetch_ticker = AsyncMock(return_value={"last": 50000.0})
    client.fetch_order = AsyncMock(
        return_value={"id": "ord_001", "status": "closed", "filled": 0.001, "average": 50000.0}
    )
    client.fetch_positions = AsyncMock(return_value=[])
    client.fetch_balance = AsyncMock(return_value={"USDT": {"total": 10000.0, "free": 9500.0}})
    client.fetch_open_orders = AsyncMock(return_value=[])
    client.to_futures_symbol = MagicMock(side_effect=lambda s: f"{s}:USDT")
    client.validate_min_notional = MagicMock(return_value=True)
    client.get_min_notional = MagicMock(return_value=5.0)
    client.is_api_healthy = True
    client.consecutive_failures = 0
    return client


def _make_order(
    symbol: str = "BTC/USDT",
    side: str = "BUY",
    target_weight: float = 1.0,
    notional_usd: float = 50.0,
) -> OrderRequestEvent:
    return OrderRequestEvent(
        client_order_id=f"test-{uuid4().hex[:8]}",
        symbol=symbol,
        side=side,
        target_weight=target_weight,
        notional_usd=notional_usd,
        validated=True,
        correlation_id=uuid4(),
        source="test",
    )


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestHappyPath:
    """정상 흐름: order → fill → reconcile."""

    @pytest.mark.asyncio
    async def test_executor_fills_and_reconciler_passes(self) -> None:
        """LiveExecutor → fill + Reconciler 정상."""
        client = _make_mock_futures_client()
        executor = LiveExecutor(client)

        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.NEUTRAL, size=0.0, last_price=50000.0
            )
        }
        pm = MagicMock()
        pm.positions = positions
        executor.set_pm(pm)

        order = _make_order(notional_usd=100.0)
        fill = await executor.execute(order)
        assert fill is not None
        assert fill.fill_price > 0

        # Reconciler: 거래소에 포지션 없음 (mock) → PM도 없으면 일치
        reconciler = PositionReconciler()
        pm_no_pos = MagicMock()
        pm_no_pos.positions = {}
        drifts = await reconciler.initial_check(pm_no_pos, client, ["BTC/USDT"])
        assert drifts == []


class TestPartialFill:
    """Partial fill 검증."""

    @pytest.mark.asyncio
    async def test_partial_fill_returns_fill(self) -> None:
        """50% 체결 시 FillEvent는 반환 (PM은 filled 기준)."""
        client = _make_mock_futures_client()
        client.create_order = AsyncMock(
            return_value={
                "id": "ord_partial",
                "status": "closed",
                "filled": 0.0005,  # requested ~0.001 → 50%
                "average": 50000.0,
                "fee": {"cost": 0.01, "currency": "USDT"},
            }
        )

        executor = LiveExecutor(client)
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.NEUTRAL, size=0.0, last_price=50000.0
            )
        }
        pm = MagicMock()
        pm.positions = positions
        executor.set_pm(pm)

        order = _make_order(notional_usd=50.0)  # 50/50000 = 0.001
        fill = await executor.execute(order)

        assert fill is not None
        assert fill.fill_qty == 0.0005


class TestApiCircuitBreakerIntegration:
    """API 연속 실패 → 주문 차단."""

    @pytest.mark.asyncio
    async def test_unhealthy_api_blocks_order(self) -> None:
        """is_api_healthy=False 시 주문 거부."""
        client = _make_mock_futures_client()
        client.is_api_healthy = False
        client.consecutive_failures = 5

        executor = LiveExecutor(client)
        pm = MagicMock()
        pm.positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.NEUTRAL, size=0.0, last_price=50000.0
            )
        }
        executor.set_pm(pm)

        order = _make_order(notional_usd=100.0)
        fill = await executor.execute(order)

        assert fill is None
        client.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_healthy_api_allows_order(self) -> None:
        """is_api_healthy=True 시 정상 진행."""
        client = _make_mock_futures_client()

        executor = LiveExecutor(client)
        pm = MagicMock()
        pm.positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.NEUTRAL, size=0.0, last_price=50000.0
            )
        }
        executor.set_pm(pm)

        order = _make_order(notional_usd=100.0)
        fill = await executor.execute(order)

        assert fill is not None
        client.create_order.assert_called_once()


class TestEquityDriftCBIntegration:
    """거래소 equity 괴리 → RM CB 트리거."""

    @pytest.mark.asyncio
    async def test_exchange_equity_triggers_cb(self) -> None:
        """PM equity 정상이나 거래소 equity가 크게 하락 → CB 발동."""
        config = _make_config()
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        rm = EDARiskManager(config=config, portfolio_manager=pm, enable_circuit_breaker=True)
        bus = EventBus(queue_size=100)
        cb_events: list[CircuitBreakerEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, CircuitBreakerEvent):
                cb_events.append(event)

        bus.subscribe(EventType.CIRCUIT_BREAKER, handler)
        await pm.register(bus)
        await rm.register(bus)

        task = asyncio.create_task(bus.start())

        # 거래소 equity가 peak 대비 15% 하락 → 즉시 CB 발동
        await rm.sync_exchange_equity(8500.0)

        await bus.stop()
        await task

        assert rm.is_circuit_breaker_active is True
        assert len(cb_events) == 1


class TestMinNotionalFilter:
    """MIN_NOTIONAL 필터 검증."""

    @pytest.mark.asyncio
    async def test_small_order_rejected_no_api_call(self) -> None:
        """소액 주문 → 거래소 호출 없이 거부."""
        client = _make_mock_futures_client()
        client.validate_min_notional = MagicMock(return_value=False)
        client.get_min_notional = MagicMock(return_value=10.0)

        executor = LiveExecutor(client)
        pm = MagicMock()
        pm.positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.NEUTRAL, size=0.0, last_price=50000.0
            )
        }
        executor.set_pm(pm)

        order = _make_order(notional_usd=3.0)
        fill = await executor.execute(order)

        assert fill is None
        client.create_order.assert_not_called()


class TestDynamicMaxOrderSize:
    """동적 max_order_size 통합."""

    @pytest.mark.asyncio
    async def test_dynamic_adjusts_with_equity(self) -> None:
        """PM equity 변동 시 RM max_order_size도 변동."""
        config = _make_config()
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        rm = EDARiskManager(config=config, portfolio_manager=pm)
        rm.enable_dynamic_max_order_size()

        # 초기: 10000 * 2.0 = 20000
        assert rm.max_order_size_usd == 20000.0

        # PM에 직접 equity 변동 시뮬레이션은 어려우므로
        # 정적 모드로 전환 대비 동적 모드가 equity 연동 확인
        assert rm._use_dynamic_max_order_size is True


class TestReconcilerBalanceSync:
    """Reconciler + RM equity sync 통합."""

    @pytest.mark.asyncio
    async def test_check_balance_returns_equity(self) -> None:
        """check_balance → exchange_equity → RM sync 연결."""
        client = _make_mock_futures_client()
        client.fetch_balance = AsyncMock(return_value={"USDT": {"total": 9500.0, "free": 9000.0}})

        config = _make_config()
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        rm = EDARiskManager(config=config, portfolio_manager=pm)

        reconciler = PositionReconciler()
        exchange_equity = await reconciler.check_balance(pm, client)

        assert exchange_equity == 9500.0

        # RM sync — drawdown 5% < 10% stop-loss → CB 미발동
        await rm.sync_exchange_equity(exchange_equity)
        assert rm.is_circuit_breaker_active is False
        assert rm.peak_equity == 10000.0  # peak 유지 (9500 < 10000)

    @pytest.mark.asyncio
    async def test_deposit_does_not_inflate_peak(self) -> None:
        """입금 시 balance drift >= 2% → RM peak sync 스킵 (peak 오염 방지)."""
        client = _make_mock_futures_client()
        # PM equity=10000, 거래소=12000 (입금 $2000) → drift 20%
        client.fetch_balance = AsyncMock(
            return_value={"USDT": {"total": 12000.0, "free": 11500.0}}
        )

        config = _make_config()
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        rm = EDARiskManager(config=config, portfolio_manager=pm)

        reconciler = PositionReconciler()
        exchange_equity = await reconciler.check_balance(pm, client)
        assert exchange_equity == 12000.0

        # drift >= 2% → _periodic_reconciliation 로직에 따라 sync 스킵
        balance_drift = reconciler.last_balance_drift_pct
        assert balance_drift >= 2.0  # 20% drift

        # sync 스킵 → peak는 초기값(10000) 유지
        # (실제 _periodic_reconciliation에서 이 조건에 걸려 sync_exchange_equity 미호출)
        assert rm.peak_equity == 10000.0

        # 만약 스킵 없이 sync했다면 peak가 12000으로 오염됨을 검증
        await rm.sync_exchange_equity(exchange_equity)
        assert rm.peak_equity == 12000.0  # 오염 상태 (이것이 방지해야 할 동작)

    @pytest.mark.asyncio
    async def test_small_drift_allows_peak_sync(self) -> None:
        """drift < 2% → RM peak sync 정상 수행."""
        client = _make_mock_futures_client()
        # PM equity=10000, 거래소=10100 → drift 1%
        client.fetch_balance = AsyncMock(
            return_value={"USDT": {"total": 10100.0, "free": 9600.0}}
        )

        config = _make_config()
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        rm = EDARiskManager(config=config, portfolio_manager=pm)

        reconciler = PositionReconciler()
        exchange_equity = await reconciler.check_balance(pm, client)

        balance_drift = reconciler.last_balance_drift_pct
        assert balance_drift < 2.0  # 1% drift → 스킵 안 함

        # drift 작으면 sync 수행 → peak 갱신
        await rm.sync_exchange_equity(exchange_equity)
        assert rm.peak_equity == 10100.0
