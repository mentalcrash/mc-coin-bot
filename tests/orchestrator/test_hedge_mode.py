"""Tests for Hedge Mode — Pod별 독립 포지션 시그널 발행.

netting_mode="hedge" 설정 시:
- Orchestrator가 Pod별 독립 시그널을 pod_id 태그와 함께 발행
- PM이 (pod_id, symbol) 복합 키로 독립 포지션 추적
- 동일 심볼에 대한 반대 방향 포지션이 가능
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pandas as pd
import pytest

from src.core.event_bus import EventBus
from src.core.events import (
    AnyEvent,
    BarEvent,
    EventType,
    FillEvent,
    OrderRequestEvent,
    SignalEvent,
)
from src.eda.executors import BacktestExecutor
from src.eda.portfolio_manager import EDAPortfolioManager
from src.models.types import Direction
from src.orchestrator.allocator import CapitalAllocator
from src.orchestrator.config import OrchestratorConfig, PodConfig
from src.orchestrator.models import AllocationMethod
from src.orchestrator.orchestrator import StrategyOrchestrator
from src.orchestrator.pod import StrategyPod
from src.portfolio.config import PortfolioManagerConfig
from src.portfolio.cost_model import CostModel
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals

# ── Test Strategy ─────────────────────────────────────────────


class LongOnlyStrategy(BaseStrategy):
    """항상 LONG(+1), strength=0.5."""

    @property
    def name(self) -> str:
        return "long_only"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        direction = pd.Series(1, index=df.index)
        strength = pd.Series(0.5, index=df.index)
        entries = pd.Series(True, index=df.index)
        exits = pd.Series(False, index=df.index)
        return StrategySignals(entries=entries, exits=exits, direction=direction, strength=strength)


class ShortOnlyStrategy(BaseStrategy):
    """항상 SHORT(-1), strength=0.3."""

    @property
    def name(self) -> str:
        return "short_only"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        direction = pd.Series(-1, index=df.index)
        strength = pd.Series(0.3, index=df.index)
        entries = pd.Series(True, index=df.index)
        exits = pd.Series(False, index=df.index)
        return StrategySignals(entries=entries, exits=exits, direction=direction, strength=strength)


# ── Helpers ───────────────────────────────────────────────────


def _make_pod_config(
    pod_id: str,
    symbols: tuple[str, ...],
    **overrides: object,
) -> PodConfig:
    defaults: dict[str, object] = {
        "pod_id": pod_id,
        "strategy_name": "test",
        "symbols": symbols,
        "initial_fraction": 0.50,
        "max_fraction": 0.50,
        "min_fraction": 0.02,
    }
    defaults.update(overrides)
    return PodConfig(**defaults)  # type: ignore[arg-type]


def _make_orch_config(
    pod_configs: tuple[PodConfig, ...],
    netting_mode: str = "hedge",
    **overrides: object,
) -> OrchestratorConfig:
    defaults: dict[str, object] = {
        "pods": pod_configs,
        "allocation_method": AllocationMethod.EQUAL_WEIGHT,
        "netting_mode": netting_mode,
        "rebalance_calendar_days": 999,
        "max_gross_leverage": 3.0,
    }
    defaults.update(overrides)
    return OrchestratorConfig(**defaults)  # type: ignore[arg-type]


def _make_bar(
    symbol: str = "ETH/USDT",
    open_: float = 100.0,
    close: float = 110.0,
    ts: datetime | None = None,
) -> BarEvent:
    if ts is None:
        ts = datetime(2024, 1, 1, tzinfo=UTC)
    return BarEvent(
        symbol=symbol,
        timeframe="1D",
        open=open_,
        high=max(open_, close) * 1.01,
        low=min(open_, close) * 0.99,
        close=close,
        volume=1000.0,
        bar_timestamp=ts,
    )


def _make_pm_config() -> PortfolioManagerConfig:
    return PortfolioManagerConfig(
        max_leverage_cap=3.0,
        system_stop_loss=0.10,
        use_trailing_stop=False,
        rebalance_threshold=0.01,
        cash_sharing=True,
        cost_model=CostModel(taker_fee=0.0004, maker_fee=0.0004, slippage=0.0),
    )


# ── Test: Event pod_id Field ──────────────────────────────────


class TestEventPodId:
    """이벤트 모델의 pod_id 필드 테스트."""

    def test_signal_event_pod_id_default_none(self) -> None:
        signal = SignalEvent(
            symbol="BTC/USDT",
            strategy_name="test",
            direction=Direction.LONG,
            strength=0.5,
            bar_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        )
        assert signal.pod_id is None

    def test_signal_event_with_pod_id(self) -> None:
        signal = SignalEvent(
            symbol="BTC/USDT",
            strategy_name="test",
            direction=Direction.LONG,
            strength=0.5,
            bar_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            pod_id="pod-a",
        )
        assert signal.pod_id == "pod-a"

    def test_order_request_pod_id(self) -> None:
        order = OrderRequestEvent(
            client_order_id="test-1",
            symbol="BTC/USDT",
            side="BUY",
            target_weight=0.5,
            notional_usd=1000.0,
            pod_id="pod-a",
        )
        assert order.pod_id == "pod-a"

    def test_fill_event_pod_id(self) -> None:
        fill = FillEvent(
            client_order_id="test-1",
            symbol="BTC/USDT",
            side="BUY",
            fill_price=50000.0,
            fill_qty=0.02,
            fee=0.5,
            fill_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            pod_id="pod-b",
        )
        assert fill.pod_id == "pod-b"


# ── Test: OrchestratorConfig netting_mode ─────────────────────


class TestNettingModeConfig:
    def test_default_signal_mode(self) -> None:
        config = _make_orch_config(
            pod_configs=(
                _make_pod_config("a", ("BTC/USDT",)),
                _make_pod_config("b", ("ETH/USDT",)),
            ),
            netting_mode="signal",
        )
        assert config.netting_mode == "signal"

    def test_hedge_mode(self) -> None:
        config = _make_orch_config(
            pod_configs=(
                _make_pod_config("a", ("BTC/USDT",)),
                _make_pod_config("b", ("ETH/USDT",)),
            ),
            netting_mode="hedge",
        )
        assert config.netting_mode == "hedge"


# ── Test: Orchestrator Hedge Signal Emission ──────────────────


class TestOrchestratorHedgeSignals:
    """Hedge mode에서 Orchestrator가 per-pod 시그널을 pod_id와 함께 발행하는지 검증."""

    @pytest.fixture
    def _setup(self) -> tuple[StrategyOrchestrator, list[SignalEvent], EventBus]:
        """Hedge mode orchestrator with 2 pods sharing ETH/USDT."""
        pod_a_config = _make_pod_config("pod-long", ("ETH/USDT",))
        pod_b_config = _make_pod_config("pod-short", ("ETH/USDT",))

        orch_config = _make_orch_config(
            pod_configs=(pod_a_config, pod_b_config),
            netting_mode="hedge",
        )

        strategy_long = LongOnlyStrategy()
        strategy_short = ShortOnlyStrategy()

        pod_a = StrategyPod(config=pod_a_config, strategy=strategy_long, capital_fraction=0.5)
        pod_a._warmup = 1
        pod_b = StrategyPod(config=pod_b_config, strategy=strategy_short, capital_fraction=0.5)
        pod_b._warmup = 1

        allocator = CapitalAllocator(config=orch_config)
        orchestrator = StrategyOrchestrator(
            config=orch_config,
            pods=[pod_a, pod_b],
            allocator=allocator,
            target_timeframe="1D",
        )

        collected_signals: list[SignalEvent] = []

        async def signal_collector(event: AnyEvent) -> None:
            assert isinstance(event, SignalEvent)
            collected_signals.append(event)

        bus = EventBus(queue_size=100)
        bus.subscribe(EventType.SIGNAL, signal_collector)

        return orchestrator, collected_signals, bus

    @pytest.mark.asyncio
    async def test_hedge_emits_per_pod_signals(
        self, _setup: tuple[StrategyOrchestrator, list[SignalEvent], EventBus]
    ) -> None:
        """Hedge mode: 동일 심볼에 대해 Pod별 독립 시그널 발행."""
        orch, signals, bus = _setup
        await orch.register(bus)

        bus_task = asyncio.create_task(bus.start())

        # Warmup bar
        bar1 = _make_bar("ETH/USDT", 100.0, 110.0, datetime(2024, 1, 1, tzinfo=UTC))
        await bus.publish(bar1)
        await bus.flush()

        # Signal bar
        bar2 = _make_bar("ETH/USDT", 110.0, 120.0, datetime(2024, 1, 2, tzinfo=UTC))
        await bus.publish(bar2)
        await bus.flush()

        # Flush remaining
        await orch.flush_pending_signals()
        await bus.flush()

        await bus.stop()
        await bus_task

        # 2개의 독립 시그널 발행 확인 (pod-long + pod-short)
        assert len(signals) >= 2

        pod_ids = {s.pod_id for s in signals}
        assert "pod-long" in pod_ids
        assert "pod-short" in pod_ids

        # 각 시그널이 올바른 pod_id를 가지는지 확인
        long_signals = [s for s in signals if s.pod_id == "pod-long"]
        short_signals = [s for s in signals if s.pod_id == "pod-short"]

        assert len(long_signals) >= 1
        assert len(short_signals) >= 1

        # Long pod → LONG direction
        assert long_signals[-1].direction == Direction.LONG
        # Short pod → SHORT direction
        assert short_signals[-1].direction == Direction.SHORT

    @pytest.mark.asyncio
    async def test_signal_mode_no_pod_id(
        self,
    ) -> None:
        """Signal mode (기존): 넷팅된 시그널에 pod_id 없음."""
        pod_a_config = _make_pod_config("pod-a", ("ETH/USDT",))
        pod_b_config = _make_pod_config("pod-b", ("ETH/USDT",))

        orch_config = _make_orch_config(
            pod_configs=(pod_a_config, pod_b_config),
            netting_mode="signal",
        )

        strategy = LongOnlyStrategy()
        pod_a = StrategyPod(config=pod_a_config, strategy=strategy, capital_fraction=0.5)
        pod_a._warmup = 1
        pod_b = StrategyPod(config=pod_b_config, strategy=strategy, capital_fraction=0.5)
        pod_b._warmup = 1

        allocator = CapitalAllocator(config=orch_config)
        orchestrator = StrategyOrchestrator(
            config=orch_config,
            pods=[pod_a, pod_b],
            allocator=allocator,
            target_timeframe="1D",
        )

        collected_signals: list[SignalEvent] = []

        async def signal_collector(event: AnyEvent) -> None:
            assert isinstance(event, SignalEvent)
            collected_signals.append(event)

        bus = EventBus(queue_size=100)
        bus.subscribe(EventType.SIGNAL, signal_collector)
        await orchestrator.register(bus)

        bus_task = asyncio.create_task(bus.start())

        bar1 = _make_bar("ETH/USDT", 100.0, 110.0, datetime(2024, 1, 1, tzinfo=UTC))
        await bus.publish(bar1)
        await bus.flush()

        bar2 = _make_bar("ETH/USDT", 110.0, 120.0, datetime(2024, 1, 2, tzinfo=UTC))
        await bus.publish(bar2)
        await bus.flush()

        await orchestrator.flush_pending_signals()
        await bus.flush()

        await bus.stop()
        await bus_task

        # Signal mode: 넷팅된 단일 시그널 (pod_id 없음)
        assert len(collected_signals) >= 1
        for sig in collected_signals:
            assert sig.pod_id is None


# ── Test: PM Hedge Mode Position Tracking ─────────────────────


class TestPMHedgeMode:
    """PM hedge mode에서 per-pod 독립 포지션 추적 검증."""

    def test_pos_key_hedge_mode(self) -> None:
        """Hedge mode에서 composite key 생성."""
        pm = EDAPortfolioManager(
            config=_make_pm_config(),
            initial_capital=10000.0,
            hedge_mode=True,
        )
        assert pm._pos_key("ETH/USDT", "pod-a") == "pod-a|ETH/USDT"
        assert pm._pos_key("ETH/USDT", None) == "ETH/USDT"

    def test_pos_key_normal_mode(self) -> None:
        """Normal mode에서 plain symbol key."""
        pm = EDAPortfolioManager(
            config=_make_pm_config(),
            initial_capital=10000.0,
            hedge_mode=False,
        )
        assert pm._pos_key("ETH/USDT", "pod-a") == "ETH/USDT"
        assert pm._pos_key("ETH/USDT", None) == "ETH/USDT"

    def test_symbol_from_key(self) -> None:
        assert EDAPortfolioManager._symbol_from_key("pod-a|ETH/USDT") == "ETH/USDT"
        assert EDAPortfolioManager._symbol_from_key("ETH/USDT") == "ETH/USDT"

    def test_pod_id_from_key(self) -> None:
        assert EDAPortfolioManager._pod_id_from_key("pod-a|ETH/USDT") == "pod-a"
        assert EDAPortfolioManager._pod_id_from_key("ETH/USDT") is None

    @pytest.mark.asyncio
    async def test_hedge_mode_independent_positions(self) -> None:
        """Hedge mode PM: 동일 심볼 반대 방향 독립 포지션."""
        pm_config = _make_pm_config()
        pm = EDAPortfolioManager(
            config=pm_config,
            initial_capital=100_000.0,
            asset_weights={"ETH/USDT": 1.0},
            target_timeframe="1D",
            hedge_mode=True,
        )

        bus = EventBus(queue_size=100)
        orders: list[OrderRequestEvent] = []

        async def order_collector(event: AnyEvent) -> None:
            assert isinstance(event, OrderRequestEvent)
            orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, order_collector)
        await pm.register(bus)
        bus_task = asyncio.create_task(bus.start())

        # Pod-a: LONG ETH 0.3
        signal_a = SignalEvent(
            symbol="ETH/USDT",
            strategy_name="test",
            direction=Direction.LONG,
            strength=0.3,
            bar_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            pod_id="pod-a",
        )
        await bus.publish(signal_a)
        await bus.flush()

        # Pod-b: SHORT ETH 0.2
        signal_b = SignalEvent(
            symbol="ETH/USDT",
            strategy_name="test",
            direction=Direction.SHORT,
            strength=0.2,
            bar_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            pod_id="pod-b",
        )
        await bus.publish(signal_b)
        await bus.flush()

        # Flush batch
        await pm.flush_pending_signals()
        await bus.flush()

        await bus.stop()
        await bus_task

        # 2개의 독립 주문 생성: BUY + SELL for ETH/USDT
        assert len(orders) == 2

        buy_orders = [o for o in orders if o.side == "BUY"]
        sell_orders = [o for o in orders if o.side == "SELL"]
        assert len(buy_orders) == 1
        assert len(sell_orders) == 1

        # pod_id 전달 확인
        assert buy_orders[0].pod_id == "pod-a"
        assert sell_orders[0].pod_id == "pod-b"

        # symbol은 실제 심볼
        assert buy_orders[0].symbol == "ETH/USDT"
        assert sell_orders[0].symbol == "ETH/USDT"

    @pytest.mark.asyncio
    async def test_hedge_mode_fill_creates_separate_positions(self) -> None:
        """Hedge mode: fill이 pod별 독립 포지션을 생성."""
        pm = EDAPortfolioManager(
            config=_make_pm_config(),
            initial_capital=100_000.0,
            asset_weights={"ETH/USDT": 1.0},
            target_timeframe="1D",
            hedge_mode=True,
        )

        bus = EventBus(queue_size=100)
        await pm.register(bus)
        bus_task = asyncio.create_task(bus.start())

        # Pod-a: BUY fill
        fill_a = FillEvent(
            client_order_id="batch-ETH/USDT-1",
            symbol="ETH/USDT",
            side="BUY",
            fill_price=3000.0,
            fill_qty=10.0,
            fee=12.0,
            fill_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            pod_id="pod-a",
        )
        await bus.publish(fill_a)
        await bus.flush()

        # Pod-b: SELL fill
        fill_b = FillEvent(
            client_order_id="batch-ETH/USDT-2",
            symbol="ETH/USDT",
            side="SELL",
            fill_price=3000.0,
            fill_qty=5.0,
            fee=6.0,
            fill_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            pod_id="pod-b",
        )
        await bus.publish(fill_b)
        await bus.flush()

        await bus.stop()
        await bus_task

        # 2개의 독립 포지션 확인
        assert "pod-a|ETH/USDT" in pm.positions
        assert "pod-b|ETH/USDT" in pm.positions

        pos_a = pm.positions["pod-a|ETH/USDT"]
        pos_b = pm.positions["pod-b|ETH/USDT"]

        assert pos_a.direction == Direction.LONG
        assert pos_a.size == pytest.approx(10.0)

        assert pos_b.direction == Direction.SHORT
        assert pos_b.size == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_hedge_mode_bar_updates_all_positions(self) -> None:
        """Hedge mode: bar가 동일 심볼 모든 포지션을 MTM 업데이트."""
        pm = EDAPortfolioManager(
            config=_make_pm_config(),
            initial_capital=100_000.0,
            asset_weights={"ETH/USDT": 1.0},
            target_timeframe="1D",
            hedge_mode=True,
        )

        bus = EventBus(queue_size=100)
        await pm.register(bus)
        bus_task = asyncio.create_task(bus.start())

        # Fill 2 independent positions
        fill_a = FillEvent(
            client_order_id="t-1",
            symbol="ETH/USDT",
            side="BUY",
            fill_price=3000.0,
            fill_qty=10.0,
            fee=0.0,
            fill_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            pod_id="pod-a",
        )
        fill_b = FillEvent(
            client_order_id="t-2",
            symbol="ETH/USDT",
            side="SELL",
            fill_price=3000.0,
            fill_qty=5.0,
            fee=0.0,
            fill_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            pod_id="pod-b",
        )
        await bus.publish(fill_a)
        await bus.flush()
        await bus.publish(fill_b)
        await bus.flush()

        # Price moves to 3300 (+10%)
        bar = BarEvent(
            symbol="ETH/USDT",
            timeframe="1D",
            open=3000.0,
            high=3300.0,
            low=2900.0,
            close=3300.0,
            volume=1000.0,
            bar_timestamp=datetime(2024, 1, 2, tzinfo=UTC),
        )
        await bus.publish(bar)
        await bus.flush()

        await bus.stop()
        await bus_task

        pos_a = pm.positions["pod-a|ETH/USDT"]
        pos_b = pm.positions["pod-b|ETH/USDT"]

        # Long: unrealized = (3300 - 3000) * 10 = +3000
        assert pos_a.unrealized_pnl == pytest.approx(3000.0)
        assert pos_a.last_price == pytest.approx(3300.0)

        # Short: unrealized = (3000 - 3300) * 5 = -1500
        assert pos_b.unrealized_pnl == pytest.approx(-1500.0)
        assert pos_b.last_price == pytest.approx(3300.0)


# ── Test: BacktestExecutor pod_id Passthrough ─────────────────


class TestExecutorPodIdPassthrough:
    """Executor가 order의 pod_id를 fill에 전달하는지 검증."""

    def test_backtest_executor_passes_pod_id(self) -> None:
        cost_model = CostModel(taker_fee=0.0004, maker_fee=0.0004, slippage=0.0)
        executor = BacktestExecutor(cost_model=cost_model)

        order = OrderRequestEvent(
            client_order_id="test-1",
            symbol="ETH/USDT",
            side="BUY",
            target_weight=0.3,
            notional_usd=3000.0,
            price=3000.0,  # SL/TS → 즉시 체결
            pod_id="pod-a",
        )

        import asyncio

        fill = asyncio.get_event_loop().run_until_complete(executor.execute(order))
        assert fill is not None
        assert fill.pod_id == "pod-a"
        assert fill.symbol == "ETH/USDT"

    def test_backtest_executor_none_pod_id(self) -> None:
        cost_model = CostModel(taker_fee=0.0004, maker_fee=0.0004, slippage=0.0)
        executor = BacktestExecutor(cost_model=cost_model)

        order = OrderRequestEvent(
            client_order_id="test-2",
            symbol="BTC/USDT",
            side="BUY",
            target_weight=0.5,
            notional_usd=5000.0,
            price=50000.0,
        )

        import asyncio

        fill = asyncio.get_event_loop().run_until_complete(executor.execute(order))
        assert fill is not None
        assert fill.pod_id is None


# ── Test: Orchestrator fill attribution in hedge mode ─────────


class TestOrchestratorHedgeFillAttribution:
    """Hedge mode에서 fill 수수료가 직접 귀속되는지 검증."""

    @pytest.mark.asyncio
    async def test_hedge_fill_direct_attribution(self) -> None:
        """Hedge mode fill → pod_id에 직접 수수료 귀속."""
        pod_config_a = _make_pod_config("pod-a", ("ETH/USDT",))
        pod_config_b = _make_pod_config("pod-b", ("ETH/USDT",))

        orch_config = _make_orch_config(
            pod_configs=(pod_config_a, pod_config_b),
            netting_mode="hedge",
        )

        strategy = LongOnlyStrategy()
        pod_a = StrategyPod(config=pod_config_a, strategy=strategy, capital_fraction=0.5)
        pod_b = StrategyPod(config=pod_config_b, strategy=strategy, capital_fraction=0.5)

        allocator = CapitalAllocator(config=orch_config)
        orchestrator = StrategyOrchestrator(
            config=orch_config,
            pods=[pod_a, pod_b],
            allocator=allocator,
            target_timeframe="1D",
        )

        # Pod-a에 가상 포지션 설정 (attribute_fee가 동작하려면 positions 필요)
        from src.orchestrator.models import PodPosition

        pod_a._positions["ETH/USDT"] = PodPosition(
            pod_id="pod-a",
            symbol="ETH/USDT",
            target_weight=0.5,
            global_weight=0.25,
            notional_usd=5000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            avg_entry_price=3000.0,
            quantity=1.67,
        )

        bus = EventBus(queue_size=100)
        await orchestrator.register(bus)
        bus_task = asyncio.create_task(bus.start())

        # Pod-a에 대한 fill (fee=10.0)
        fill = FillEvent(
            client_order_id="test-1",
            symbol="ETH/USDT",
            side="BUY",
            fill_price=3000.0,
            fill_qty=1.0,
            fee=10.0,
            fill_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            pod_id="pod-a",
        )
        await bus.publish(fill)
        await bus.flush()

        await bus.stop()
        await bus_task

        # Pod-a: fee 차감 → realized_pnl = 0.0 - 10.0 = -10.0
        pos_a = pod_a._positions.get("ETH/USDT")
        assert pos_a is not None
        assert pos_a.realized_pnl == pytest.approx(-10.0)

        # Pod-b: 수수료 없음 (포지션 없으므로 변경 없음)
        pos_b = pod_b._positions.get("ETH/USDT")
        assert pos_b is None
