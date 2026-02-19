"""Tests for Orchestrator State Persistence — Phase 9.

PageHinkley/Pod/Lifecycle/Orchestrator 직렬화 + E2E save→restore.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pandas as pd
import pytest

from src.eda.persistence.database import Database
from src.eda.persistence.state_manager import StateManager
from src.orchestrator.allocator import CapitalAllocator
from src.orchestrator.config import (
    GraduationCriteria,
    OrchestratorConfig,
    PodConfig,
    RetirementCriteria,
)
from src.orchestrator.degradation import PageHinkleyDetector
from src.orchestrator.lifecycle import LifecycleManager
from src.orchestrator.models import AllocationMethod, LifecycleState, PodPosition
from src.orchestrator.orchestrator import StrategyOrchestrator
from src.orchestrator.pod import StrategyPod
from src.orchestrator.state_persistence import (
    _KEY_DAILY_RETURNS,
    _KEY_HISTORIES,
    _KEY_STATE,
    _MAX_ALLOCATION_HISTORY,
    _MAX_DAILY_RETURNS,
    OrchestratorStatePersistence,
)
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals

# ── Test Strategy ─────────────────────────────────────────────────


class SimpleTestStrategy(BaseStrategy):
    """close > open → LONG(+1), else SHORT(-1)."""

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
        return StrategySignals(entries=entries, exits=exits, direction=direction, strength=strength)


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


def _make_pod(
    pod_id: str = "pod-a",
    symbols: tuple[str, ...] = ("BTC/USDT",),
    capital_fraction: float = 0.25,
) -> StrategyPod:
    config = _make_pod_config(pod_id=pod_id, symbols=symbols)
    strategy = SimpleTestStrategy()
    pod = StrategyPod(config=config, strategy=strategy, capital_fraction=capital_fraction)
    pod._warmup = 3
    return pod


def _make_orchestrator_config(
    pod_configs: tuple[PodConfig, ...] | None = None,
) -> OrchestratorConfig:
    if pod_configs is None:
        pod_configs = (
            _make_pod_config("pod-a", ("BTC/USDT",)),
            _make_pod_config("pod-b", ("ETH/USDT",)),
        )
    return OrchestratorConfig(
        pods=pod_configs,
        allocation_method=AllocationMethod.EQUAL_WEIGHT,
    )


def _make_orchestrator(
    config: OrchestratorConfig | None = None,
    lifecycle: LifecycleManager | None = None,
) -> StrategyOrchestrator:
    if config is None:
        config = _make_orchestrator_config()
    pods = [_make_pod(pod_id=pc.pod_id, symbols=pc.symbols) for pc in config.pods]
    allocator = CapitalAllocator(config)
    return StrategyOrchestrator(
        config=config,
        pods=pods,
        allocator=allocator,
        lifecycle_manager=lifecycle,
    )


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    database = Database(":memory:")
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def state_manager(db: Database) -> StateManager:
    return StateManager(db)


@pytest.fixture
async def persistence(state_manager: StateManager) -> OrchestratorStatePersistence:
    return OrchestratorStatePersistence(state_manager)


# ══════════════════════════════════════════════════════════════════
# 1. PageHinkleyDetector Serialization
# ══════════════════════════════════════════════════════════════════


class TestPageHinkleyDetectorSerialization:
    def test_round_trip(self) -> None:
        """to_dict → restore_from_dict preserves state."""
        ph = PageHinkleyDetector(delta=0.01, lambda_=30.0, alpha=0.999)
        for v in [0.01, -0.02, 0.03, -0.01, 0.005]:
            ph.update(v)

        data = ph.to_dict()
        ph2 = PageHinkleyDetector(delta=0.01, lambda_=30.0, alpha=0.999)
        ph2.restore_from_dict(data)

        assert ph2.n_observations == ph.n_observations
        assert ph2.score == pytest.approx(ph.score)

    def test_score_preserved_after_restore(self) -> None:
        """Restored detector produces identical scores on new data."""
        ph = PageHinkleyDetector()
        for v in [0.01, -0.005, 0.02]:
            ph.update(v)

        data = ph.to_dict()
        ph2 = PageHinkleyDetector()
        ph2.restore_from_dict(data)

        # Feed identical new value to both
        result1 = ph.update(0.015)
        result2 = ph2.update(0.015)
        assert result1 == result2
        assert ph.score == pytest.approx(ph2.score)

    def test_missing_fields_use_defaults(self) -> None:
        """Partial dict → remaining fields default to 0."""
        ph = PageHinkleyDetector()
        ph.restore_from_dict({"n": 5})  # only 'n' provided
        assert ph.n_observations == 5
        assert ph.score == pytest.approx(0.0)

    def test_config_not_in_dict(self) -> None:
        """to_dict excludes config params (delta, lambda_, alpha)."""
        ph = PageHinkleyDetector(delta=0.123, lambda_=99.9, alpha=0.5)
        data = ph.to_dict()
        assert "delta" not in data
        assert "lambda_" not in data
        assert "alpha" not in data
        assert set(data.keys()) == {"n", "x_mean", "m_t", "m_min"}


# ══════════════════════════════════════════════════════════════════
# 2. StrategyPod Serialization
# ══════════════════════════════════════════════════════════════════


class TestPodSerialization:
    def test_state_round_trip(self) -> None:
        pod = _make_pod()
        pod.state = LifecycleState.PRODUCTION
        data = pod.to_dict()

        pod2 = _make_pod()
        pod2.restore_from_dict(data)
        assert pod2.state == LifecycleState.PRODUCTION

    def test_capital_fraction_round_trip(self) -> None:
        pod = _make_pod(capital_fraction=0.35)
        data = pod.to_dict()

        pod2 = _make_pod(capital_fraction=0.10)
        pod2.restore_from_dict(data)
        assert pod2.capital_fraction == pytest.approx(0.35)

    def test_target_weights_round_trip(self) -> None:
        pod = _make_pod()
        pod._target_weights = {"BTC/USDT": 0.8, "ETH/USDT": -0.3}
        data = pod.to_dict()

        pod2 = _make_pod()
        pod2.restore_from_dict(data)
        assert pod2.get_target_weights() == {
            "BTC/USDT": pytest.approx(0.8),
            "ETH/USDT": pytest.approx(-0.3),
        }

    def test_positions_round_trip(self) -> None:
        pod = _make_pod()
        pod._positions["BTC/USDT"] = PodPosition(
            pod_id="pod-a",
            symbol="BTC/USDT",
            notional_usd=5000.0,
            realized_pnl=100.0,
            unrealized_pnl=-20.0,
        )
        data = pod.to_dict()

        pod2 = _make_pod()
        pod2.restore_from_dict(data)
        pos = pod2._positions["BTC/USDT"]
        assert pos.notional_usd == pytest.approx(5000.0)
        assert pos.realized_pnl == pytest.approx(100.0)
        assert pos.unrealized_pnl == pytest.approx(-20.0)

    def test_performance_round_trip(self) -> None:
        pod = _make_pod()
        perf = pod.performance
        perf.total_return = 0.15
        perf.sharpe_ratio = 1.5
        perf.max_drawdown = 0.08
        perf.calmar_ratio = 2.0
        perf.trade_count = 42
        perf.live_days = 90
        perf.peak_equity = 11000.0
        perf.current_equity = 10800.0

        data = pod.to_dict()
        pod2 = _make_pod()
        pod2.restore_from_dict(data)

        p2 = pod2.performance
        assert p2.total_return == pytest.approx(0.15)
        assert p2.sharpe_ratio == pytest.approx(1.5)
        assert p2.max_drawdown == pytest.approx(0.08)
        assert p2.calmar_ratio == pytest.approx(2.0)
        assert p2.trade_count == 42
        assert p2.live_days == 90
        assert p2.peak_equity == pytest.approx(11000.0)

    def test_performance_last_updated_round_trip(self) -> None:
        pod = _make_pod()
        ts = datetime(2026, 2, 1, 12, 0, 0, tzinfo=UTC)
        pod.performance.last_updated = ts
        data = pod.to_dict()

        pod2 = _make_pod()
        pod2.restore_from_dict(data)
        assert pod2.performance.last_updated == ts

    def test_partial_restore_missing_keys(self) -> None:
        """Partial dict — only 'state' provided, rest defaults."""
        pod = _make_pod()
        pod.restore_from_dict({"state": "production"})
        assert pod.state == LifecycleState.PRODUCTION
        # Others unchanged from defaults
        assert pod.capital_fraction == pytest.approx(0.25)

    def test_empty_dict_is_noop(self) -> None:
        pod = _make_pod()
        original_state = pod.state
        original_fraction = pod.capital_fraction
        pod.restore_from_dict({})
        assert pod.state == original_state
        assert pod.capital_fraction == pytest.approx(original_fraction)


# ══════════════════════════════════════════════════════════════════
# 3. LifecycleManager Serialization
# ══════════════════════════════════════════════════════════════════


class TestLifecycleSerialization:
    def _make_lifecycle(self) -> LifecycleManager:
        return LifecycleManager(
            graduation=GraduationCriteria(),
            retirement=RetirementCriteria(),
        )

    def test_empty_state_round_trip(self) -> None:
        lm = self._make_lifecycle()
        data = lm.to_dict()
        assert data == {}

    def test_pod_state_round_trip(self) -> None:
        lm = self._make_lifecycle()
        pod = _make_pod()
        lm.evaluate(pod)  # creates internal state

        data = lm.to_dict()
        assert "pod-a" in data

        lm2 = self._make_lifecycle()
        lm2.restore_from_dict(data)
        assert "pod-a" in lm2._pod_states

    def test_ph_detector_continuity(self) -> None:
        """PH detector state survives serialization."""
        lm = self._make_lifecycle()
        pod = _make_pod()
        lm.evaluate(pod)

        # Feed some data to PH detector
        pls = lm._pod_states["pod-a"]
        for v in [0.01, -0.02, 0.03]:
            pls.ph_detector.update(v)
        original_score = pls.ph_detector.score

        data = lm.to_dict()
        lm2 = self._make_lifecycle()
        lm2.restore_from_dict(data)

        pls2 = lm2._pod_states["pod-a"]
        assert pls2.ph_detector.score == pytest.approx(original_score)

    def test_state_entered_at_preserved(self) -> None:
        lm = self._make_lifecycle()
        pod = _make_pod()
        lm.evaluate(pod)

        pls = lm._pod_states["pod-a"]
        original_entered = pls.state_entered_at

        data = lm.to_dict()
        lm2 = self._make_lifecycle()
        lm2.restore_from_dict(data)

        pls2 = lm2._pod_states["pod-a"]
        assert pls2.state_entered_at == original_entered

    def test_unmatched_pod_id_created(self) -> None:
        """Restore with unknown pod_id creates new state entry."""
        lm = self._make_lifecycle()
        data = {"unknown-pod": {"consecutive_loss_months": 3, "last_monthly_check_day": 2}}
        lm.restore_from_dict(data)
        assert "unknown-pod" in lm._pod_states
        assert lm._pod_states["unknown-pod"].consecutive_loss_months == 3


# ══════════════════════════════════════════════════════════════════
# 4. StrategyOrchestrator Serialization
# ══════════════════════════════════════════════════════════════════


class TestOrchestratorSerialization:
    def test_rebalance_ts_round_trip(self) -> None:
        orch = _make_orchestrator()
        ts = datetime(2026, 2, 10, 8, 0, 0, tzinfo=UTC)
        orch._last_rebalance_ts = ts
        data = orch.to_dict()

        orch2 = _make_orchestrator()
        orch2.restore_from_dict(data)
        assert orch2._last_rebalance_ts == ts

    def test_rebalance_ts_none(self) -> None:
        orch = _make_orchestrator()
        orch._last_rebalance_ts = None
        data = orch.to_dict()

        orch2 = _make_orchestrator()
        orch2._last_rebalance_ts = datetime(2026, 1, 1, tzinfo=UTC)
        orch2.restore_from_dict(data)
        assert orch2._last_rebalance_ts is None

    def test_pod_targets_round_trip(self) -> None:
        orch = _make_orchestrator()
        orch._last_pod_targets = {
            "pod-a": {"BTC/USDT": 0.5},
            "pod-b": {"ETH/USDT": -0.3},
        }
        data = orch.to_dict()

        orch2 = _make_orchestrator()
        orch2.restore_from_dict(data)
        targets = orch2._last_pod_targets
        assert targets["pod-a"]["BTC/USDT"] == pytest.approx(0.5)
        assert targets["pod-b"]["ETH/USDT"] == pytest.approx(-0.3)

    def test_empty_dict_restore(self) -> None:
        orch = _make_orchestrator()
        orch.restore_from_dict({})
        assert orch._last_rebalance_ts is None


# ══════════════════════════════════════════════════════════════════
# 5. E2E Persistence (OrchestratorStatePersistence)
# ══════════════════════════════════════════════════════════════════


class TestE2EPersistence:
    @pytest.mark.asyncio
    async def test_full_save_restore(self, persistence: OrchestratorStatePersistence) -> None:
        """Full round-trip: save → new orchestrator → restore."""
        orch = _make_orchestrator()
        # Set some state
        orch.pods[0].state = LifecycleState.PRODUCTION
        orch.pods[0].capital_fraction = 0.6
        orch.pods[0].record_daily_return(0.01)
        orch.pods[0].record_daily_return(-0.005)
        orch.pods[1].record_daily_return(0.02)
        ts = datetime(2026, 2, 10, tzinfo=UTC)
        orch._last_rebalance_ts = ts

        await persistence.save(orch)

        # New orchestrator (fresh)
        orch2 = _make_orchestrator()
        result = await persistence.restore(orch2)

        assert result is True
        assert orch2.pods[0].state == LifecycleState.PRODUCTION
        assert orch2.pods[0].capital_fraction == pytest.approx(0.6)
        assert len(orch2.pods[0].daily_returns) == 2
        assert orch2.pods[0].daily_returns[0] == pytest.approx(0.01)
        assert len(orch2.pods[1].daily_returns) == 1
        assert orch2._last_rebalance_ts == ts

    @pytest.mark.asyncio
    async def test_no_saved_state_returns_false(
        self, persistence: OrchestratorStatePersistence
    ) -> None:
        orch = _make_orchestrator()
        result = await persistence.restore(orch)
        assert result is False

    @pytest.mark.asyncio
    async def test_corrupted_json_returns_false(
        self, persistence: OrchestratorStatePersistence
    ) -> None:
        await persistence._save_key(_KEY_STATE, "{invalid json!!!")

        orch = _make_orchestrator()
        result = await persistence.restore(orch)
        assert result is False

    @pytest.mark.asyncio
    async def test_future_version_returns_false(
        self, persistence: OrchestratorStatePersistence
    ) -> None:
        import json

        await persistence._save_key(_KEY_STATE, json.dumps({"version": 999}))

        orch = _make_orchestrator()
        result = await persistence.restore(orch)
        assert result is False

    @pytest.mark.asyncio
    async def test_pod_added_gets_defaults(self, persistence: OrchestratorStatePersistence) -> None:
        """Pod added in config after save → starts with defaults."""
        # Save with 1 pod
        config1 = _make_orchestrator_config(pod_configs=(_make_pod_config("pod-a", ("BTC/USDT",)),))
        orch1 = _make_orchestrator(config=config1)
        orch1.pods[0].state = LifecycleState.PRODUCTION
        await persistence.save(orch1)

        # Restore with 2 pods (pod-b is new)
        orch2 = _make_orchestrator()
        result = await persistence.restore(orch2)
        assert result is True
        assert orch2.pods[0].state == LifecycleState.PRODUCTION  # restored
        assert orch2.pods[1].state == LifecycleState.INCUBATION  # default

    @pytest.mark.asyncio
    async def test_pod_removed_ignored(self, persistence: OrchestratorStatePersistence) -> None:
        """Pod removed from config → saved state ignored."""
        orch1 = _make_orchestrator()
        orch1.pods[0].state = LifecycleState.PRODUCTION
        orch1.pods[1].state = LifecycleState.WARNING
        await persistence.save(orch1)

        # Restore with only 1 pod
        config2 = _make_orchestrator_config(pod_configs=(_make_pod_config("pod-a", ("BTC/USDT",)),))
        orch2 = _make_orchestrator(config=config2)
        result = await persistence.restore(orch2)
        assert result is True
        assert orch2.pods[0].state == LifecycleState.PRODUCTION

    @pytest.mark.asyncio
    async def test_daily_returns_trim_270(self, persistence: OrchestratorStatePersistence) -> None:
        """daily_returns > 270 trimmed on save."""
        config = _make_orchestrator_config(pod_configs=(_make_pod_config("pod-a", ("BTC/USDT",)),))
        orch = _make_orchestrator(config=config)
        for i in range(300):
            orch.pods[0].record_daily_return(0.001 * (i + 1))

        await persistence.save(orch)

        orch2 = _make_orchestrator(config=config)
        await persistence.restore(orch2)
        assert len(orch2.pods[0].daily_returns) == _MAX_DAILY_RETURNS
        # Should have last 270 values (31st to 300th)
        assert orch2.pods[0].daily_returns[0] == pytest.approx(0.001 * 31)

    @pytest.mark.asyncio
    async def test_live_days_synced_with_daily_returns(
        self, persistence: OrchestratorStatePersistence
    ) -> None:
        config = _make_orchestrator_config(pod_configs=(_make_pod_config("pod-a", ("BTC/USDT",)),))
        orch = _make_orchestrator(config=config)
        for _ in range(50):
            orch.pods[0].record_daily_return(0.01)

        await persistence.save(orch)

        orch2 = _make_orchestrator(config=config)
        await persistence.restore(orch2)
        assert orch2.pods[0].performance.live_days == 50

    @pytest.mark.asyncio
    async def test_lifecycle_persistence(self, persistence: OrchestratorStatePersistence) -> None:
        """Lifecycle manager state persisted and restored."""
        lifecycle = LifecycleManager(
            graduation=GraduationCriteria(),
            retirement=RetirementCriteria(),
        )
        orch = _make_orchestrator(lifecycle=lifecycle)

        # Trigger lifecycle evaluation to create internal state
        lifecycle.evaluate(orch.pods[0])
        lifecycle.evaluate(orch.pods[1])

        # Feed PH detector
        pls = lifecycle._pod_states["pod-a"]
        for v in [0.01, -0.02, 0.03]:
            pls.ph_detector.update(v)
        original_score = pls.ph_detector.score

        await persistence.save(orch)

        lifecycle2 = LifecycleManager(
            graduation=GraduationCriteria(),
            retirement=RetirementCriteria(),
        )
        orch2 = _make_orchestrator(lifecycle=lifecycle2)
        await persistence.restore(orch2)

        assert "pod-a" in lifecycle2._pod_states
        assert lifecycle2._pod_states["pod-a"].ph_detector.score == pytest.approx(original_score)

    @pytest.mark.asyncio
    async def test_lifecycle_none_skipped(self, persistence: OrchestratorStatePersistence) -> None:
        """Orchestrator without lifecycle → save/restore works."""
        orch = _make_orchestrator(lifecycle=None)
        orch.pods[0].state = LifecycleState.PRODUCTION
        await persistence.save(orch)

        orch2 = _make_orchestrator(lifecycle=None)
        result = await persistence.restore(orch2)
        assert result is True
        assert orch2.pods[0].state == LifecycleState.PRODUCTION

    @pytest.mark.asyncio
    async def test_positions_persist(self, persistence: OrchestratorStatePersistence) -> None:
        config = _make_orchestrator_config(pod_configs=(_make_pod_config("pod-a", ("BTC/USDT",)),))
        orch = _make_orchestrator(config=config)
        orch.pods[0].update_position("BTC/USDT", 0.1, 50000.0, 5.0, is_buy=True)
        await persistence.save(orch)

        orch2 = _make_orchestrator(config=config)
        await persistence.restore(orch2)
        pos = orch2.pods[0]._positions["BTC/USDT"]
        assert pos.notional_usd == pytest.approx(0.1 * 50000.0)

    @pytest.mark.asyncio
    async def test_atomic_save_both_keys(self, persistence: OrchestratorStatePersistence) -> None:
        """H-6: save()가 두 key를 단일 트랜잭션으로 저장."""
        orch = _make_orchestrator()
        orch.pods[0].state = LifecycleState.PRODUCTION
        orch.pods[0].record_daily_return(0.01)

        await persistence.save(orch)

        # 두 key 모두 존재하는지 확인
        state_raw = await persistence._load_key(_KEY_STATE)
        returns_raw = await persistence._load_key(_KEY_DAILY_RETURNS)
        assert state_raw is not None
        assert returns_raw is not None

        # 복원 가능 확인
        orch2 = _make_orchestrator()
        result = await persistence.restore(orch2)
        assert result is True
        assert orch2.pods[0].state == LifecycleState.PRODUCTION
        assert len(orch2.pods[0].daily_returns) == 1

    @pytest.mark.asyncio
    async def test_corrupted_daily_returns_skipped(
        self, persistence: OrchestratorStatePersistence
    ) -> None:
        """Corrupted daily_returns key → state still restored."""
        orch = _make_orchestrator()
        orch.pods[0].state = LifecycleState.PRODUCTION
        await persistence.save(orch)

        # Corrupt daily_returns key
        await persistence._save_key(_KEY_DAILY_RETURNS, "not valid json!")

        orch2 = _make_orchestrator()
        result = await persistence.restore(orch2)
        assert result is True
        assert orch2.pods[0].state == LifecycleState.PRODUCTION
        assert len(orch2.pods[0].daily_returns) == 0  # daily_returns not restored


# ══════════════════════════════════════════════════════════════════
# 6. Histories Persistence (Step 1)
# ══════════════════════════════════════════════════════════════════


class TestHistoriesPersistence:
    @pytest.mark.asyncio
    async def test_histories_save_restore_round_trip(
        self, persistence: OrchestratorStatePersistence
    ) -> None:
        """allocation/lifecycle/risk histories 왕복 저장."""
        orch = _make_orchestrator()
        orch._allocation_history.append({"timestamp": "2026-02-01", "pod-a": 0.5})
        orch._lifecycle_events.append(
            {"pod_id": "pod-a", "from_state": "incubation", "to_state": "production"}
        )
        orch._risk_contributions_history.append({"timestamp": "2026-02-01", "pod-a": 0.5})

        await persistence.save(orch)

        orch2 = _make_orchestrator()
        await persistence.restore(orch2)

        assert len(orch2.allocation_history) == 1
        assert orch2.allocation_history[0]["pod-a"] == 0.5
        assert len(orch2.lifecycle_events) == 1
        assert orch2.lifecycle_events[0]["pod_id"] == "pod-a"
        assert len(orch2.risk_contributions_history) == 1

    @pytest.mark.asyncio
    async def test_histories_trim_on_save(self, persistence: OrchestratorStatePersistence) -> None:
        """500건 초과 → 최근 500건만 저장."""
        orch = _make_orchestrator()
        for i in range(600):
            orch._allocation_history.append({"timestamp": f"day-{i}", "value": i})

        await persistence.save(orch)

        orch2 = _make_orchestrator()
        await persistence.restore(orch2)

        assert len(orch2.allocation_history) == _MAX_ALLOCATION_HISTORY
        # 마지막 500건 (100~599)
        assert orch2.allocation_history[0]["value"] == 100

    @pytest.mark.asyncio
    async def test_histories_missing_key_noop(
        self, persistence: OrchestratorStatePersistence
    ) -> None:
        """키 없으면 빈 리스트 유지."""
        # Save without histories key (only state + daily_returns)
        orch = _make_orchestrator()
        await persistence.save(orch)

        # Delete histories key
        conn = persistence._db.connection
        await conn.execute("DELETE FROM bot_state WHERE key = ?", (_KEY_HISTORIES,))
        await conn.commit()

        orch2 = _make_orchestrator()
        await persistence.restore(orch2)

        assert len(orch2.allocation_history) == 0
        assert len(orch2.lifecycle_events) == 0
        assert len(orch2.risk_contributions_history) == 0

    @pytest.mark.asyncio
    async def test_histories_corrupted_json_skipped(
        self, persistence: OrchestratorStatePersistence
    ) -> None:
        """파싱 실패 → 경고 + 빈 리스트."""
        orch = _make_orchestrator()
        orch._allocation_history.append({"test": True})
        await persistence.save(orch)

        # Corrupt histories key
        await persistence._save_key(_KEY_HISTORIES, "invalid json{{{")

        orch2 = _make_orchestrator()
        result = await persistence.restore(orch2)
        assert result is True
        assert len(orch2.allocation_history) == 0
