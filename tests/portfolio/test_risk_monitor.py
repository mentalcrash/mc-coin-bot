"""PortfolioRiskMonitor 단위 테스트."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.core.event_bus import EventBus
from src.core.events import (
    BalanceUpdateEvent,
    BarEvent,
    CircuitBreakerEvent,
    EventType,
    FillEvent,
    RiskAlertEvent,
)
from src.portfolio.risk_monitor import PortfolioRiskMonitor
from src.portfolio.risk_monitor_models import (
    PortfolioRiskConfig,
    RiskAction,
)

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def bus() -> EventBus:
    return EventBus(queue_size=100)


@pytest.fixture
def config() -> PortfolioRiskConfig:
    return PortfolioRiskConfig(
        max_portfolio_drawdown=0.20,
        max_daily_loss=0.05,
        max_correlation_exposure=0.70,
        max_concentration_pct=0.40,
    )


@pytest.fixture
def monitor(config: PortfolioRiskConfig) -> PortfolioRiskMonitor:
    return PortfolioRiskMonitor(config=config)


# ── Helper ────────────────────────────────────────────────────────


def _make_balance_event(equity: float, drawdown_pct: float = 0.0) -> BalanceUpdateEvent:
    return BalanceUpdateEvent(
        total_equity=equity,
        available_cash=equity * 0.5,
        drawdown_pct=drawdown_pct,
        open_position_count=1,
    )


def _make_fill_event(symbol: str, side: str, price: float, qty: float) -> FillEvent:
    return FillEvent(
        client_order_id="test-001",
        symbol=symbol,
        side=side,
        fill_price=price,
        fill_qty=qty,
        fill_timestamp=datetime.now(UTC),
    )


def _make_bar_event(symbol: str, close: float) -> BarEvent:
    return BarEvent(
        symbol=symbol,
        timeframe="1D",
        open=close,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=1000.0,
        bar_timestamp=datetime.now(UTC),
    )


async def _drain_bus_events(bus: EventBus) -> list[object]:
    """bus queue에 쌓인 이벤트를 drain하여 반환."""
    events: list[object] = []
    while not bus._queue.empty():
        try:
            evt = bus._queue.get_nowait()
            if evt is not None:
                events.append(evt)
        except asyncio.QueueEmpty:
            break
    return events


# ── Tests: Models ─────────────────────────────────────────────────


class TestRiskModels:
    """RiskAction, PortfolioRiskConfig 모델 테스트."""

    def test_risk_action_values(self) -> None:
        assert RiskAction.NORMAL == "normal"
        assert RiskAction.LIQUIDATE_ALL == "liquidate_all"
        assert RiskAction.HALT_NEW_ENTRIES == "halt_new_entries"
        assert RiskAction.REDUCE_CORRELATED == "reduce_correlated"

    def test_config_defaults(self) -> None:
        config = PortfolioRiskConfig()
        assert config.max_portfolio_drawdown == 0.20
        assert config.max_daily_loss == 0.05
        assert config.max_correlation_exposure == 0.70
        assert config.max_concentration_pct == 0.40

    def test_config_frozen(self) -> None:
        config = PortfolioRiskConfig()
        with pytest.raises(ValidationError):
            config.max_portfolio_drawdown = 0.30  # type: ignore[misc]


# ── Tests: Registration ──────────────────────────────────────────


class TestRegistration:
    """EventBus 등록 테스트."""

    async def test_register(self, bus: EventBus, monitor: PortfolioRiskMonitor) -> None:
        await monitor.register(bus)
        assert monitor._bus is bus
        assert len(bus._handlers[EventType.BALANCE_UPDATE]) > 0
        assert len(bus._handlers[EventType.FILL]) > 0
        assert len(bus._handlers[EventType.BAR]) > 0

    def test_set_initial_equity(self, monitor: PortfolioRiskMonitor) -> None:
        monitor.set_initial_equity(10000.0)
        assert monitor._initial_equity == 10000.0
        assert monitor._peak_equity == 10000.0
        assert monitor._current_equity == 10000.0


# ── Tests: Balance Updates ────────────────────────────────────────


class TestBalanceUpdates:
    """BALANCE_UPDATE 처리 테스트."""

    async def test_normal_balance(self, bus: EventBus, monitor: PortfolioRiskMonitor) -> None:
        """정상 범위 내 balance → NORMAL."""
        await monitor.register(bus)
        monitor.set_initial_equity(10000.0)

        event = _make_balance_event(9500.0)
        await monitor._on_balance_update(event)

        snapshot = monitor.last_snapshot
        assert snapshot is not None
        assert snapshot.action == RiskAction.NORMAL

    async def test_drawdown_triggers_liquidation(
        self, bus: EventBus, monitor: PortfolioRiskMonitor
    ) -> None:
        """20% drawdown → LIQUIDATE_ALL + CircuitBreakerEvent in bus queue."""
        await monitor.register(bus)
        monitor.set_initial_equity(10000.0)

        event = _make_balance_event(7900.0)  # -21% drawdown
        await monitor._on_balance_update(event)

        snapshot = monitor.last_snapshot
        assert snapshot is not None
        assert snapshot.action == RiskAction.LIQUIDATE_ALL

        # bus queue에 CircuitBreakerEvent가 존재하는지 확인
        queued_events = await _drain_bus_events(bus)
        cb_events = [e for e in queued_events if isinstance(e, CircuitBreakerEvent)]
        assert len(cb_events) == 1
        assert cb_events[0].close_all_positions is True

    async def test_drawdown_warning(self, bus: EventBus, monitor: PortfolioRiskMonitor) -> None:
        """16~20% drawdown → WARNING RiskAlertEvent in bus queue."""
        await monitor.register(bus)
        monitor.set_initial_equity(10000.0)

        event = _make_balance_event(8300.0)  # -17% drawdown (> 80% of 20%)
        await monitor._on_balance_update(event)

        queued_events = await _drain_bus_events(bus)
        risk_alerts = [e for e in queued_events if isinstance(e, RiskAlertEvent)]
        assert len(risk_alerts) > 0
        assert any("approaching" in a.message for a in risk_alerts)

    async def test_peak_equity_tracking(self, bus: EventBus, monitor: PortfolioRiskMonitor) -> None:
        """Equity 신고점 갱신."""
        await monitor.register(bus)
        monitor.set_initial_equity(10000.0)

        await monitor._on_balance_update(_make_balance_event(11000.0))
        assert monitor._peak_equity == 11000.0

        await monitor._on_balance_update(_make_balance_event(10500.0))
        assert monitor._peak_equity == 11000.0  # 하락해도 peak 유지


# ── Tests: Daily Loss ─────────────────────────────────────────────


class TestDailyLoss:
    """일일 손실 한도 테스트."""

    async def test_daily_loss_halt(self, bus: EventBus) -> None:
        """5% 일일 손실 → HALT_NEW_ENTRIES + RiskAlertEvent."""
        config = PortfolioRiskConfig(max_daily_loss=0.05, max_portfolio_drawdown=0.50)
        monitor = PortfolioRiskMonitor(config=config)
        await monitor.register(bus)
        monitor.set_initial_equity(10000.0)

        # 당일 시작 설정
        monitor._day_start_equity = 10000.0
        monitor._current_day = datetime.now(UTC).timetuple().tm_yday

        event = _make_balance_event(9400.0)  # -6% daily loss
        await monitor._on_balance_update(event)

        snapshot = monitor.last_snapshot
        assert snapshot is not None
        assert snapshot.action == RiskAction.HALT_NEW_ENTRIES

        queued_events = await _drain_bus_events(bus)
        risk_alerts = [e for e in queued_events if isinstance(e, RiskAlertEvent)]
        assert len(risk_alerts) == 1


# ── Tests: Concentration ──────────────────────────────────────────


class TestConcentration:
    """단일 심볼 집중도 테스트."""

    async def test_concentration_tracking(
        self, bus: EventBus, monitor: PortfolioRiskMonitor
    ) -> None:
        """FILL 이벤트로 심볼별 notional 추적."""
        await monitor.register(bus)
        monitor.set_initial_equity(10000.0)

        fill1 = _make_fill_event("BTC/USDT", "BUY", 50000.0, 0.1)
        fill2 = _make_fill_event("ETH/USDT", "BUY", 3000.0, 1.0)
        await monitor._on_fill(fill1)
        await monitor._on_fill(fill2)

        # BTC: 5000, ETH: 3000 → BTC 집중도 = 5000/8000 = 0.625
        concentration = monitor._compute_max_concentration()
        assert concentration == pytest.approx(5000.0 / 8000.0, abs=0.01)


# ── Tests: Correlation ────────────────────────────────────────────


class TestCorrelation:
    """상관 모니터링 테스트."""

    async def test_no_correlation_with_insufficient_data(
        self, monitor: PortfolioRiskMonitor
    ) -> None:
        """데이터 부족 시 상관 0.0 반환."""
        assert monitor._compute_avg_correlation() == 0.0

    async def test_correlation_with_price_history(
        self, bus: EventBus, monitor: PortfolioRiskMonitor
    ) -> None:
        """가격 히스토리 축적 후 상관 계산."""
        await monitor.register(bus)

        # 동일 방향 가격 이동 → 높은 상관
        for i in range(10):
            price = 50000 + i * 100
            await monitor._on_bar(_make_bar_event("BTC/USDT", price))
            await monitor._on_bar(_make_bar_event("ETH/USDT", price * 0.06))

        corr = monitor._compute_avg_correlation()
        assert corr > 0.5

    async def test_bar_history_capped(self, bus: EventBus, monitor: PortfolioRiskMonitor) -> None:
        """가격 히스토리 100개 제한."""
        await monitor.register(bus)

        for i in range(150):
            await monitor._on_bar(_make_bar_event("BTC/USDT", 50000 + i))

        assert len(monitor._price_history["BTC/USDT"]) == 100


# ── Tests: Combined Scenarios ─────────────────────────────────────


class TestCombinedScenarios:
    """복합 시나리오 테스트."""

    async def test_liquidation_overrides_halt(self, bus: EventBus) -> None:
        """LIQUIDATE_ALL은 HALT_NEW_ENTRIES보다 우선."""
        config = PortfolioRiskConfig(
            max_portfolio_drawdown=0.20,
            max_daily_loss=0.05,
        )
        monitor = PortfolioRiskMonitor(config=config)
        await monitor.register(bus)
        monitor.set_initial_equity(10000.0)
        monitor._day_start_equity = 10000.0
        monitor._current_day = datetime.now(UTC).timetuple().tm_yday

        # 25% drawdown + 25% daily loss → LIQUIDATE_ALL
        event = _make_balance_event(7500.0)
        await monitor._on_balance_update(event)

        snapshot = monitor.last_snapshot
        assert snapshot is not None
        assert snapshot.action == RiskAction.LIQUIDATE_ALL

    async def test_no_alerts_when_normal(
        self, bus: EventBus, monitor: PortfolioRiskMonitor
    ) -> None:
        """정상 범위 → 알림 없음."""
        await monitor.register(bus)
        monitor.set_initial_equity(10000.0)

        event = _make_balance_event(9800.0)  # -2% (정상 범위)
        await monitor._on_balance_update(event)

        queued_events = await _drain_bus_events(bus)
        assert len(queued_events) == 0

    async def test_default_config(self, bus: EventBus) -> None:
        """기본 설정으로 생성 가능."""
        monitor = PortfolioRiskMonitor()
        await monitor.register(bus)
        assert monitor._config.max_portfolio_drawdown == 0.20
