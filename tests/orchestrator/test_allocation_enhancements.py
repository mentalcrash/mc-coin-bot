"""Tests for Orchestrator Allocation Enhancements (E1-E7).

E1: 3-Layer Leverage Defense (per-symbol, per-pod, portfolio)
E2: All-Excluded Signal Suppression (should_emit_signals)
E3: Rolling Metrics (30-day rolling Sharpe/DD)
E4: Absolute Eligibility Thresholds (absolute_min_sharpe, absolute_max_drawdown)
E5: Risk Defense Turnover Awareness (config flag)
E6: Allocation Dashboard
E7: Retired Pod Target Cleanup
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pandas as pd
import pytest

from src.core.event_bus import EventBus
from src.core.events import AnyEvent, BarEvent, EventType, SignalEvent
from src.models.types import Direction
from src.orchestrator.allocator import CapitalAllocator
from src.orchestrator.asset_selector import AssetSelector
from src.orchestrator.config import (
    AssetSelectorConfig,
    OrchestratorConfig,
    PodConfig,
)
from src.orchestrator.dashboard import AllocationDashboard
from src.orchestrator.models import AllocationMethod, AssetLifecycleState, LifecycleState
from src.orchestrator.orchestrator import StrategyOrchestrator
from src.orchestrator.pod import _ROLLING_WINDOW, StrategyPod
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals

# ── Test Strategy ─────────────────────────────────────────────────


class SimpleTestStrategy(BaseStrategy):
    """close > open → LONG(+1), else SHORT(-1). strength = abs(close-open)/open."""

    @property
    def name(self) -> str:
        return "test_simple"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        direction = (df["close"] > df["open"]).astype(int) * 2 - 1
        strength = ((df["close"] - df["open"]).abs() / df["open"]).shift(1).fillna(0.01)
        entries = direction.diff().fillna(0).abs() > 0
        exits = pd.Series(False, index=df.index)
        return StrategySignals(
            entries=entries,
            exits=exits,
            direction=direction,
            strength=strength,
        )


class HighStrengthStrategy(BaseStrategy):
    """항상 LONG(+1) strength=3.0 반환 (레버리지 테스트용)."""

    @property
    def name(self) -> str:
        return "test_high_strength"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        direction = pd.Series(1, index=df.index)
        strength = pd.Series(3.0, index=df.index)
        entries = pd.Series(True, index=df.index)
        exits = pd.Series(False, index=df.index)
        return StrategySignals(
            entries=entries,
            exits=exits,
            direction=direction,
            strength=strength,
        )


# ── Helpers ───────────────────────────────────────────────────────


def _make_pod_config(
    pod_id: str = "pod-a",
    symbols: tuple[str, ...] = ("BTC/USDT",),
    **overrides: object,
) -> PodConfig:
    defaults: dict[str, object] = {
        "pod_id": pod_id,
        "strategy_name": "tsmom",
        "symbols": symbols,
        "initial_fraction": 0.10,
        "max_fraction": 0.40,
        "min_fraction": 0.02,
    }
    defaults.update(overrides)
    return PodConfig(**defaults)  # type: ignore[arg-type]


def _make_orchestrator_config(
    pod_configs: tuple[PodConfig, ...] | None = None,
    **overrides: object,
) -> OrchestratorConfig:
    if pod_configs is None:
        pod_configs = (
            _make_pod_config("pod-a", ("BTC/USDT",)),
            _make_pod_config("pod-b", ("ETH/USDT",)),
        )
    defaults: dict[str, object] = {
        "pods": pod_configs,
        "allocation_method": AllocationMethod.EQUAL_WEIGHT,
        "rebalance_calendar_days": 7,
    }
    defaults.update(overrides)
    return OrchestratorConfig(**defaults)  # type: ignore[arg-type]


def _make_pod(
    pod_id: str = "pod-a",
    symbols: tuple[str, ...] = ("BTC/USDT",),
    capital_fraction: float = 0.5,
    warmup: int = 3,
    strategy: BaseStrategy | None = None,
    **pod_overrides: object,
) -> StrategyPod:
    config = _make_pod_config(pod_id, symbols, **pod_overrides)
    strat = strategy or SimpleTestStrategy()
    pod = StrategyPod(config=config, strategy=strat, capital_fraction=capital_fraction)
    pod._warmup = warmup
    return pod


def _make_bar(
    symbol: str = "BTC/USDT",
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
        correlation_id=uuid4(),
        source="test",
    )


def _feed_warmup_bars(
    pod: StrategyPod,
    symbol: str,
    n: int,
    base_ts: datetime,
) -> datetime:
    ts = base_ts
    for i in range(n):
        bar_data = {
            "open": 100.0 + i,
            "high": (101.0 + i) * 1.01,
            "low": (100.0 + i) * 0.99,
            "close": 101.0 + i,
            "volume": 1000.0,
        }
        pod.compute_signal(symbol, bar_data, ts)
        ts += timedelta(days=1)
    return ts


async def _run_with_bus(
    orchestrator: StrategyOrchestrator,
    events: list[AnyEvent],
) -> list[SignalEvent]:
    bus = EventBus(queue_size=100)
    signals: list[SignalEvent] = []

    async def signal_collector(event: AnyEvent) -> None:
        assert isinstance(event, SignalEvent)
        signals.append(event)

    bus.subscribe(EventType.SIGNAL, signal_collector)
    await orchestrator.register(bus)

    task = asyncio.create_task(bus.start())

    for evt in events:
        await bus.publish(evt)
        await bus.flush()

    await orchestrator.flush_pending_signals()
    await bus.flush()

    await bus.stop()
    await task

    return signals


# ── E1: 3-Layer Leverage Defense ──────────────────────────────────


class TestE1PerSymbolLeverageCap:
    """Layer 1: compute_signal()에서 per-symbol strength cap."""

    def test_strength_capped_at_max_leverage(self) -> None:
        """Strategy strength > max_leverage → max_leverage로 클램프."""
        pod = _make_pod(
            "pod-a",
            ("BTC/USDT",),
            strategy=HighStrengthStrategy(),
            max_leverage=2.0,
        )
        pod._warmup = 3
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup_bars(pod, "BTC/USDT", 2, ts)

        bar_data = {
            "open": 100.0,
            "high": 111.0,
            "low": 99.0,
            "close": 110.0,
            "volume": 1000.0,
        }
        result = pod.compute_signal("BTC/USDT", bar_data, ts + timedelta(days=2))
        assert result is not None
        _, strength = result
        # HighStrengthStrategy returns 3.0, capped at max_leverage=2.0
        assert strength <= 2.0

    def test_strength_not_capped_when_below(self) -> None:
        """Strategy strength < max_leverage → 원본 유지."""
        pod = _make_pod(
            "pod-a",
            ("BTC/USDT",),
            max_leverage=5.0,
        )
        pod._warmup = 3
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup_bars(pod, "BTC/USDT", 2, ts)

        bar_data = {
            "open": 100.0,
            "high": 111.0,
            "low": 99.0,
            "close": 110.0,
            "volume": 1000.0,
        }
        result = pod.compute_signal("BTC/USDT", bar_data, ts + timedelta(days=2))
        assert result is not None
        _, strength = result
        # SimpleTestStrategy returns small strength, should not be capped
        assert strength > 0
        assert strength < 5.0


class TestE1PerPodLeverageCap:
    """Layer 2: Pod aggregate leverage cap in orchestrator."""

    def test_per_pod_leverage_cap_scales_down(self) -> None:
        """Pod gross > max_leverage * capital_fraction → 비례 축소."""
        pod = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=0.5, max_leverage=1.5)
        config = _make_orchestrator_config((pod.config,), max_gross_leverage=20.0)
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod], allocator)

        # Simulate pod targets exceeding leverage
        orch._last_pod_targets["pod-a"] = {"BTC/USDT": 2.0}
        # pod_limit = 1.5 * 0.5 = 0.75, pod_gross = 2.0 > 0.75

        orch._apply_per_pod_leverage_cap()

        capped_weight = orch._last_pod_targets["pod-a"]["BTC/USDT"]
        assert abs(capped_weight) <= 1.5 * 0.5 + 1e-8

    def test_per_pod_leverage_cap_no_change_when_within(self) -> None:
        """Pod gross <= limit → 변경 없음."""
        pod = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=0.5, max_leverage=5.0)
        config = _make_orchestrator_config((pod.config,))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod], allocator)

        orch._last_pod_targets["pod-a"] = {"BTC/USDT": 0.3}
        # pod_limit = 5.0 * 0.5 = 2.5, pod_gross = 0.3 < 2.5

        orch._apply_per_pod_leverage_cap()

        assert orch._last_pod_targets["pod-a"]["BTC/USDT"] == pytest.approx(0.3)

    def test_per_pod_multi_symbol_proportional_scale(self) -> None:
        """멀티 심볼 Pod → 비례적으로 축소."""
        pod = _make_pod(
            "pod-a",
            ("BTC/USDT", "ETH/USDT"),
            capital_fraction=0.5,
            max_leverage=2.0,
        )
        config = _make_orchestrator_config((pod.config,), max_gross_leverage=20.0)
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod], allocator)

        # gross = |0.6| + |0.8| = 1.4, limit = 2.0 * 0.5 = 1.0
        orch._last_pod_targets["pod-a"] = {"BTC/USDT": 0.6, "ETH/USDT": 0.8}

        orch._apply_per_pod_leverage_cap()

        targets = orch._last_pod_targets["pod-a"]
        gross_after = abs(targets["BTC/USDT"]) + abs(targets["ETH/USDT"])
        assert gross_after <= 1.0 + 1e-8
        # Proportional: BTC/ETH ratio preserved
        assert targets["BTC/USDT"] / targets["ETH/USDT"] == pytest.approx(0.6 / 0.8)


class TestE1IntegrationLeverageLayers:
    """Layer 1+2+3 통합 테스트."""

    async def test_full_leverage_pipeline(self) -> None:
        """High strength → per-symbol cap → per-pod cap → portfolio cap."""
        pod = _make_pod(
            "pod-a",
            ("BTC/USDT",),
            capital_fraction=1.0,
            strategy=HighStrengthStrategy(),
            max_leverage=2.0,
        )
        config = _make_orchestrator_config(
            (pod.config,),
            max_gross_leverage=1.5,
        )
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod], allocator)

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup_bars(pod, "BTC/USDT", 2, ts)

        bar = _make_bar("BTC/USDT", 100.0, 110.0, ts + timedelta(days=2))
        signals = await _run_with_bus(orch, [bar])

        btc_signals = [s for s in signals if s.symbol == "BTC/USDT"]
        assert len(btc_signals) == 1
        # Final strength should be <= portfolio max (1.5)
        assert btc_signals[0].strength <= 1.5 + 1e-6


# ── E2: All-Excluded Signal Suppression ───────────────────────────


class TestE2ShouldEmitSignals:
    """should_emit_signals property 테스트."""

    def test_active_pod_should_emit(self) -> None:
        pod = _make_pod()
        assert pod.should_emit_signals is True

    def test_retired_pod_should_not_emit(self) -> None:
        pod = _make_pod()
        pod.state = LifecycleState.RETIRED
        assert pod.should_emit_signals is False

    def test_paused_pod_should_not_emit(self) -> None:
        pod = _make_pod()
        pod.pause()
        assert pod.should_emit_signals is False

    def test_all_excluded_should_not_emit(self) -> None:
        """AssetSelector가 모든 에셋을 제외하면 should_emit=False."""
        asc = AssetSelectorConfig(
            enabled=True,
            exclude_score_threshold=0.20,
            include_score_threshold=0.35,
            min_active_assets=1,
            sharpe_lookback=20,
            return_lookback=10,
        )
        config = _make_pod_config(
            symbols=("BTC/USDT", "ETH/USDT"),
            asset_selector=asc,
        )
        pod = StrategyPod(
            config=config,
            strategy=SimpleTestStrategy(),
            capital_fraction=0.5,
        )
        assert pod._asset_selector is not None

        # Force all excluded
        for sym in ("BTC/USDT", "ETH/USDT"):
            pod._asset_selector._states[sym].multiplier = 0.0

        assert pod.should_emit_signals is False

    def test_partial_excluded_should_emit(self) -> None:
        """일부만 제외 → should_emit=True."""
        asc = AssetSelectorConfig(
            enabled=True,
            exclude_score_threshold=0.20,
            include_score_threshold=0.35,
            min_active_assets=1,
            sharpe_lookback=20,
            return_lookback=10,
        )
        config = _make_pod_config(
            symbols=("BTC/USDT", "ETH/USDT"),
            asset_selector=asc,
        )
        pod = StrategyPod(
            config=config,
            strategy=SimpleTestStrategy(),
            capital_fraction=0.5,
        )
        assert pod._asset_selector is not None

        pod._asset_selector._states["BTC/USDT"].multiplier = 0.0
        # ETH still active
        assert pod.should_emit_signals is True

    def test_no_selector_should_emit(self) -> None:
        """AssetSelector 없으면 should_emit=True."""
        pod = _make_pod()
        assert pod._asset_selector is None
        assert pod.should_emit_signals is True


class TestE2OrchestratorSignalSuppression:
    """All-excluded pod의 시그널이 orchestrator에서 차단되는지 검증."""

    async def test_all_excluded_pod_no_signal(self) -> None:
        """All-excluded pod → SignalEvent 미발행 (NEUTRAL만)."""
        asc = AssetSelectorConfig(
            enabled=True,
            exclude_score_threshold=0.20,
            include_score_threshold=0.35,
            min_active_assets=1,
            sharpe_lookback=20,
            return_lookback=10,
        )
        pod_cfg = _make_pod_config(
            symbols=("BTC/USDT",),
            asset_selector=asc,
        )
        pod = StrategyPod(
            config=pod_cfg,
            strategy=SimpleTestStrategy(),
            capital_fraction=1.0,
        )
        pod._warmup = 3

        config = _make_orchestrator_config((pod_cfg,))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod], allocator)

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup_bars(pod, "BTC/USDT", 2, ts)

        # Force all excluded after warmup
        assert pod._asset_selector is not None
        pod._asset_selector._states["BTC/USDT"].multiplier = 0.0

        bar = _make_bar("BTC/USDT", 100.0, 110.0, ts + timedelta(days=2))
        signals = await _run_with_bus(orch, [bar])

        # Pod computes signal internally but doesn't emit
        btc_signals = [s for s in signals if s.symbol == "BTC/USDT"]
        # Either no signal or NEUTRAL (strength=0)
        for s in btc_signals:
            assert s.direction == Direction.NEUTRAL or s.strength == pytest.approx(0.0)


# ── E3: Rolling Metrics ───────────────────────────────────────────


class TestE3RollingMetrics:
    """30-day rolling Sharpe/DD 테스트."""

    def test_rolling_sharpe_empty(self) -> None:
        pod = _make_pod()
        assert pod.rolling_sharpe == pytest.approx(0.0)

    def test_rolling_sharpe_few_returns(self) -> None:
        pod = _make_pod()
        pod.record_daily_return(0.01)
        assert pod.rolling_sharpe == pytest.approx(0.0)

    def test_rolling_sharpe_positive(self) -> None:
        pod = _make_pod()
        # 양의 평균 + 약간의 분산 필요 (vol > 0)
        for i in range(30):
            pod.record_daily_return(0.01 + 0.001 * (i % 5 - 2))
        assert pod.rolling_sharpe > 0

    def test_rolling_sharpe_window_size(self) -> None:
        """100일 기록 → 최근 30일만 사용."""
        pod = _make_pod()
        # 70일: 음의 수익률 (분산 있음)
        for i in range(70):
            pod.record_daily_return(-0.01 + 0.001 * (i % 5 - 2))
        # 30일: 양의 수익률 (분산 있음)
        for i in range(30):
            pod.record_daily_return(0.01 + 0.001 * (i % 5 - 2))
        # rolling_sharpe는 최근 30일 기준 → 양수
        assert pod.rolling_sharpe > 0

    def test_rolling_drawdown_empty(self) -> None:
        pod = _make_pod()
        assert pod.rolling_drawdown == pytest.approx(0.0)

    def test_rolling_drawdown_no_loss(self) -> None:
        pod = _make_pod()
        for _ in range(10):
            pod.record_daily_return(0.01)
        assert pod.rolling_drawdown == pytest.approx(0.0)

    def test_rolling_drawdown_with_loss(self) -> None:
        pod = _make_pod()
        pod.record_daily_return(0.10)  # +10%
        pod.record_daily_return(-0.15)  # -15%
        assert pod.rolling_drawdown > 0

    def test_rolling_window_constant(self) -> None:
        assert _ROLLING_WINDOW == 30


# ── E4: Absolute Eligibility Thresholds ───────────────────────────


class TestE4AbsoluteThresholds:
    """AssetSelector absolute Sharpe/DD threshold 테스트."""

    def _make_selector(
        self,
        symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT"),
        **overrides: object,
    ) -> AssetSelector:
        defaults: dict[str, object] = {
            "enabled": True,
            "exclude_score_threshold": 0.20,
            "include_score_threshold": 0.35,
            "exclude_confirmation_bars": 3,
            "include_confirmation_bars": 2,
            "min_exclusion_bars": 5,
            "ramp_steps": 1,
            "min_active_assets": 1,
            "sharpe_lookback": 20,
            "return_lookback": 10,
        }
        defaults.update(overrides)
        cfg = AssetSelectorConfig(**defaults)  # type: ignore[arg-type]
        return AssetSelector(config=cfg, symbols=symbols)

    def test_absolute_min_sharpe_exclusion(self) -> None:
        """Sharpe < absolute_min_sharpe → COOLDOWN."""
        selector = self._make_selector(
            absolute_min_sharpe=-0.5,
        )
        # BTC with terrible returns (Sharpe < -0.5)
        bad_returns = [-0.05 + 0.01 * (i % 3 - 1) for i in range(30)]
        returns = {
            "BTC/USDT": bad_returns,
            "ETH/USDT": [0.01 + 0.001 * (i % 5 - 2) for i in range(30)],
            "SOL/USDT": [0.01 + 0.001 * (i % 5 - 2) for i in range(30)],
        }
        closes = dict.fromkeys(returns, 100.0)

        selector.on_bar(returns, closes)

        btc_state = selector._states["BTC/USDT"].state
        assert btc_state == AssetLifecycleState.COOLDOWN

    def test_absolute_max_drawdown_exclusion(self) -> None:
        """DD > absolute_max_drawdown → COOLDOWN."""
        selector = self._make_selector(
            absolute_max_drawdown=0.05,
            sharpe_lookback=30,  # 전체 30일 lookback
        )
        # BTC: 끝부분에 큰 하락 (lookback 범위 내)
        btc_rets = [0.01] * 20 + [0.05, -0.20] + [0.01] * 8
        returns = {
            "BTC/USDT": btc_rets,
            "ETH/USDT": [0.01 + 0.001 * (i % 5 - 2) for i in range(30)],
            "SOL/USDT": [0.01 + 0.001 * (i % 5 - 2) for i in range(30)],
        }
        closes = dict.fromkeys(returns, 100.0)

        selector.on_bar(returns, closes)

        btc_state = selector._states["BTC/USDT"].state
        assert btc_state == AssetLifecycleState.COOLDOWN

    def test_no_absolute_threshold_backward_compat(self) -> None:
        """absolute thresholds=None → 기존 동작 유지."""
        selector = self._make_selector(
            absolute_min_sharpe=None,
            absolute_max_drawdown=None,
        )
        returns = {s: [0.01] * 30 for s in ("BTC/USDT", "ETH/USDT", "SOL/USDT")}
        closes = dict.fromkeys(returns, 100.0)

        for _ in range(5):
            selector.on_bar(returns, closes)

        for s in returns:
            assert selector._states[s].state == AssetLifecycleState.ACTIVE


class TestE4PermanentExclusion:
    """max_cooldown_cycles 초과 시 영구 제외 테스트."""

    def test_permanent_exclusion_after_cycles(self) -> None:
        """cooldown_cycles >= max_cooldown_cycles → permanently_excluded."""
        cfg = AssetSelectorConfig(
            enabled=True,
            exclude_score_threshold=0.20,
            include_score_threshold=0.35,
            exclude_confirmation_bars=1,
            include_confirmation_bars=1,
            ramp_steps=1,
            min_active_assets=1,
            min_exclusion_bars=1,
            sharpe_lookback=20,
            return_lookback=10,
            max_cooldown_cycles=2,
        )
        selector = AssetSelector(config=cfg, symbols=("BTC/USDT", "ETH/USDT", "SOL/USDT"))

        st = selector._states["BTC/USDT"]
        st.cooldown_cycles = 1  # Already had 1 cycle

        # Force another COOLDOWN entry
        st.state = AssetLifecycleState.ACTIVE
        st.multiplier = 1.0
        st.score = 0.10
        selector._transition("BTC/USDT")  # → UNDERPERFORMING → ramp 1 step

        # Check if UNDERPERFORMING → COOLDOWN (ramp_steps=1)
        if st.state == AssetLifecycleState.UNDERPERFORMING:
            selector._transition("BTC/USDT")  # complete ramp → COOLDOWN

        assert st.cooldown_cycles >= 2
        assert st.permanently_excluded is True

    def test_permanent_excluded_no_transition(self) -> None:
        """permanently_excluded → FSM 전이 불가."""
        cfg = AssetSelectorConfig(
            enabled=True,
            exclude_score_threshold=0.20,
            include_score_threshold=0.35,
            exclude_confirmation_bars=1,
            include_confirmation_bars=1,
            ramp_steps=1,
            min_active_assets=1,
            min_exclusion_bars=1,
            sharpe_lookback=20,
            return_lookback=10,
        )
        selector = AssetSelector(config=cfg, symbols=("BTC/USDT", "ETH/USDT"))

        st = selector._states["BTC/USDT"]
        st.permanently_excluded = True
        st.state = AssetLifecycleState.COOLDOWN
        st.multiplier = 0.0
        st.score = 0.50  # High score, should trigger RE_ENTRY normally
        st.cooldown_bars = 100

        selector._transition("BTC/USDT")

        # Still COOLDOWN (permanently excluded)
        assert st.state == AssetLifecycleState.COOLDOWN

    def test_serialization_round_trip_permanent(self) -> None:
        """permanently_excluded + cooldown_cycles 직렬화."""
        cfg = AssetSelectorConfig(
            enabled=True,
            exclude_score_threshold=0.20,
            include_score_threshold=0.35,
            sharpe_lookback=20,
            return_lookback=10,
        )
        selector = AssetSelector(config=cfg, symbols=("BTC/USDT", "ETH/USDT"))

        selector._states["BTC/USDT"].permanently_excluded = True
        selector._states["BTC/USDT"].cooldown_cycles = 3

        data = selector.to_dict()
        selector2 = AssetSelector(config=cfg, symbols=("BTC/USDT", "ETH/USDT"))
        selector2.restore_from_dict(data)

        assert selector2._states["BTC/USDT"].permanently_excluded is True
        assert selector2._states["BTC/USDT"].cooldown_cycles == 3


# ── E5: Risk Defense Turnover Awareness ───────────────────────────


class TestE5RiskDefenseTurnoverConfig:
    """risk_defense_bypass_turnover config flag 테스트."""

    def test_default_bypass_true(self) -> None:
        config = _make_orchestrator_config()
        assert config.risk_defense_bypass_turnover is True

    def test_explicit_false(self) -> None:
        config = _make_orchestrator_config(risk_defense_bypass_turnover=False)
        assert config.risk_defense_bypass_turnover is False


# ── E6: Allocation Dashboard ─────────────────────────────────────


class TestE6AllocationDashboard:
    """AllocationDashboard 기능 테스트."""

    def _make_orch(self) -> StrategyOrchestrator:
        pod_a = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=0.6)
        pod_b = _make_pod("pod-b", ("ETH/USDT",), capital_fraction=0.4)
        config = _make_orchestrator_config((pod_a.config, pod_b.config))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod_a, pod_b], allocator)
        orch._last_allocated_weights = {"pod-a": 0.5, "pod-b": 0.5}
        return orch

    def test_compute_drift(self) -> None:
        orch = self._make_orch()
        dashboard = AllocationDashboard(orch)

        snapshot = dashboard.compute_drift()
        assert len(snapshot.pod_drifts) == 2
        assert snapshot.total_drift > 0
        assert snapshot.max_drift > 0
        assert snapshot.effective_n > 0

        # pod-a: current=0.6, target=0.5, drift=0.1
        pod_a_drift = next(d for d in snapshot.pod_drifts if d.pod_id == "pod-a")
        assert pod_a_drift.drift == pytest.approx(0.1)
        assert pod_a_drift.drift_pct == pytest.approx(0.2)  # 0.1/0.5

    def test_get_timeline_empty(self) -> None:
        orch = self._make_orch()
        dashboard = AllocationDashboard(orch)

        timeline = dashboard.get_timeline()
        assert len(timeline.timestamps) == 0
        assert len(timeline.lifecycle_events) == 0

    def test_get_timeline_with_history(self) -> None:
        orch = self._make_orch()
        orch._allocation_history = [
            {"timestamp": "2024-01-01", "pod-a": 0.5, "pod-b": 0.5},
            {"timestamp": "2024-01-08", "pod-a": 0.6, "pod-b": 0.4},
        ]
        dashboard = AllocationDashboard(orch)

        timeline = dashboard.get_timeline()
        assert len(timeline.timestamps) == 2
        assert len(timeline.pod_fractions["pod-a"]) == 2
        assert timeline.pod_fractions["pod-a"][1] == pytest.approx(0.6)

    def test_get_pod_leverage_usage(self) -> None:
        orch = self._make_orch()
        orch._last_pod_targets["pod-a"] = {"BTC/USDT": 0.3}
        orch._last_pod_targets["pod-b"] = {"ETH/USDT": 0.2}

        dashboard = AllocationDashboard(orch)
        usage = dashboard.get_pod_leverage_usage()

        assert "pod-a" in usage
        assert usage["pod-a"]["gross_leverage"] == pytest.approx(0.3)
        assert usage["pod-a"]["usage_pct"] > 0


# ── E7: Retired Pod Target Cleanup ────────────────────────────────


class TestE7RetiredPodTargetCleanup:
    """Retired pod 잔여 타겟 zero-out 테스트."""

    def test_retired_targets_zeroed(self) -> None:
        pod_a = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=0.5)
        pod_b = _make_pod("pod-b", ("ETH/USDT",), capital_fraction=0.5)
        pod_b.state = LifecycleState.RETIRED

        config = _make_orchestrator_config((pod_a.config, pod_b.config))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod_a, pod_b], allocator)

        # Simulate existing targets
        orch._last_pod_targets["pod-a"] = {"BTC/USDT": 0.3}
        orch._last_pod_targets["pod-b"] = {"ETH/USDT": 0.4}

        orch._cleanup_retired_pod_targets()

        # pod-a active → 유지
        assert orch._last_pod_targets["pod-a"]["BTC/USDT"] == pytest.approx(0.3)
        # pod-b retired → zeroed
        assert orch._last_pod_targets["pod-b"]["ETH/USDT"] == pytest.approx(0.0)

    def test_no_retired_no_change(self) -> None:
        pod_a = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=1.0)

        config = _make_orchestrator_config((pod_a.config,))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod_a], allocator)

        orch._last_pod_targets["pod-a"] = {"BTC/USDT": 0.5}
        orch._cleanup_retired_pod_targets()

        assert orch._last_pod_targets["pod-a"]["BTC/USDT"] == pytest.approx(0.5)

    def test_retired_cleanup_in_rebalance(self) -> None:
        """_execute_rebalance()에서 lifecycle 평가 후 cleanup 호출."""
        pod_a = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=0.5)
        pod_b = _make_pod("pod-b", ("ETH/USDT",), capital_fraction=0.5)

        config = _make_orchestrator_config(
            (pod_a.config, pod_b.config),
            rebalance_calendar_days=1,
            min_rebalance_turnover=0.0,
        )
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod_a, pod_b], allocator)

        # Record returns for allocator
        for _ in range(5):
            pod_a.record_daily_return(0.01)
            pod_b.record_daily_return(0.01)

        # Simulate retired pod with stale targets
        pod_b.state = LifecycleState.RETIRED
        orch._last_pod_targets["pod-b"] = {"ETH/USDT": 0.3}

        orch._execute_rebalance()

        # Retired targets should be zeroed
        assert orch._last_pod_targets["pod-b"]["ETH/USDT"] == pytest.approx(0.0)


# ── E4 Config Tests ────────────────────────────────────────────────


class TestE4ConfigFields:
    """AssetSelectorConfig 새 필드 검증."""

    def test_absolute_min_sharpe_none_default(self) -> None:
        cfg = AssetSelectorConfig(enabled=True)
        assert cfg.absolute_min_sharpe is None

    def test_absolute_max_drawdown_none_default(self) -> None:
        cfg = AssetSelectorConfig(enabled=True)
        assert cfg.absolute_max_drawdown is None

    def test_max_cooldown_cycles_none_default(self) -> None:
        cfg = AssetSelectorConfig(enabled=True)
        assert cfg.max_cooldown_cycles is None

    def test_absolute_min_sharpe_set(self) -> None:
        cfg = AssetSelectorConfig(
            enabled=True,
            absolute_min_sharpe=-1.0,
        )
        assert cfg.absolute_min_sharpe == pytest.approx(-1.0)

    def test_absolute_max_drawdown_set(self) -> None:
        cfg = AssetSelectorConfig(
            enabled=True,
            absolute_max_drawdown=0.10,
        )
        assert cfg.absolute_max_drawdown == pytest.approx(0.10)

    def test_max_cooldown_cycles_set(self) -> None:
        cfg = AssetSelectorConfig(
            enabled=True,
            max_cooldown_cycles=3,
        )
        assert cfg.max_cooldown_cycles == 3
