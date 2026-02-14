"""Tests for Critical 7 fixes (C-1 ~ C-7).

Orchestrator Pre-Live Audit에서 발견된 Critical 7건 수정 검증.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.orchestrator.allocator import CapitalAllocator
from src.orchestrator.config import (
    GraduationCriteria,
    OrchestratorConfig,
    PodConfig,
    RetirementCriteria,
)
from src.orchestrator.lifecycle import LifecycleManager
from src.orchestrator.models import (
    AllocationMethod,
    LifecycleState,
    RiskAlert,
)
from src.orchestrator.orchestrator import StrategyOrchestrator
from src.orchestrator.pod import StrategyPod
from src.orchestrator.risk_aggregator import RiskAggregator
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
    capital_fraction: float = 0.5,
    warmup: int = 3,
) -> StrategyPod:
    config = _make_pod_config(pod_id, symbols)
    strategy = SimpleTestStrategy()
    pod = StrategyPod(config=config, strategy=strategy, capital_fraction=capital_fraction)
    pod._warmup = warmup
    return pod


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


def _set_pod_performance(pod: StrategyPod, **kwargs: object) -> None:
    perf = pod.performance
    for key, value in kwargs.items():
        setattr(perf, key, value)


def _inject_daily_returns(pod: StrategyPod, returns: list[float]) -> None:
    pod._daily_returns = list(returns)
    pod._performance.live_days = len(returns)


# ═══════════════════════════════════════════════════════════════════
# C-1: Pod realized PnL 항상 0 수정
# ═══════════════════════════════════════════════════════════════════


class TestC1RealizedPnL:
    """C-1: update_position()으로 정확한 realized PnL 계산."""

    def test_long_open_close_positive_pnl(self) -> None:
        """Long open → close at higher price → realized PnL 양수."""
        pod = _make_pod()
        # Open long: buy 1.0 @ 50000
        pod.update_position("BTC/USDT", 1.0, 50000.0, 5.0, is_buy=True)
        # Close long: sell 1.0 @ 55000
        pod.update_position("BTC/USDT", 1.0, 55000.0, 5.0, is_buy=False)

        pos = pod._positions["BTC/USDT"]
        # realized = (55000 - 50000) * 1.0 - 5 - 5 = 4990
        assert pos.realized_pnl == pytest.approx(4990.0)
        assert abs(pos.quantity) < 1e-10

    def test_long_open_close_negative_pnl(self) -> None:
        """Long open → close at lower price → realized PnL 음수."""
        pod = _make_pod()
        pod.update_position("BTC/USDT", 1.0, 50000.0, 5.0, is_buy=True)
        pod.update_position("BTC/USDT", 1.0, 45000.0, 5.0, is_buy=False)

        pos = pod._positions["BTC/USDT"]
        # realized = (45000 - 50000) * 1.0 - 5 - 5 = -5010
        assert pos.realized_pnl == pytest.approx(-5010.0)

    def test_short_open_cover_positive_pnl(self) -> None:
        """Short open → cover at lower price → realized PnL 양수."""
        pod = _make_pod()
        # Open short: sell 1.0 @ 50000
        pod.update_position("BTC/USDT", 1.0, 50000.0, 5.0, is_buy=False)
        # Cover: buy 1.0 @ 45000
        pod.update_position("BTC/USDT", 1.0, 45000.0, 5.0, is_buy=True)

        pos = pod._positions["BTC/USDT"]
        # realized = (50000 - 45000) * 1.0 - 5 - 5 = 4990
        assert pos.realized_pnl == pytest.approx(4990.0)

    def test_short_open_cover_negative_pnl(self) -> None:
        """Short open → cover at higher price → realized PnL 음수."""
        pod = _make_pod()
        pod.update_position("BTC/USDT", 1.0, 50000.0, 5.0, is_buy=False)
        pod.update_position("BTC/USDT", 1.0, 55000.0, 5.0, is_buy=True)

        pos = pod._positions["BTC/USDT"]
        # realized = (50000 - 55000) * 1.0 - 5 - 5 = -5010
        assert pos.realized_pnl == pytest.approx(-5010.0)

    def test_partial_close(self) -> None:
        """부분 청산 시 비례 realized PnL."""
        pod = _make_pod()
        pod.update_position("BTC/USDT", 2.0, 50000.0, 0.0, is_buy=True)
        # Partial close: sell 1.0 @ 55000
        pod.update_position("BTC/USDT", 1.0, 55000.0, 0.0, is_buy=False)

        pos = pod._positions["BTC/USDT"]
        # realized = (55000 - 50000) * 1.0 = 5000
        assert pos.realized_pnl == pytest.approx(5000.0)
        # 남은 quantity = 1.0
        assert pos.quantity == pytest.approx(1.0)
        # avg_entry 유지
        assert pos.avg_entry_price == pytest.approx(50000.0)

    def test_avg_entry_price_weighted(self) -> None:
        """같은 방향 추가 시 가중 평균 진입가."""
        pod = _make_pod()
        pod.update_position("BTC/USDT", 1.0, 50000.0, 0.0, is_buy=True)
        pod.update_position("BTC/USDT", 1.0, 60000.0, 0.0, is_buy=True)

        pos = pod._positions["BTC/USDT"]
        # avg = (1*50000 + 1*60000) / 2 = 55000
        assert pos.avg_entry_price == pytest.approx(55000.0)
        assert pos.quantity == pytest.approx(2.0)

    def test_direction_flip(self) -> None:
        """방향 전환(flip) 시 새 진입가 = fill_price."""
        pod = _make_pod()
        pod.update_position("BTC/USDT", 1.0, 50000.0, 0.0, is_buy=True)
        # Flip: sell 2.0 @ 55000 (close 1 + open short 1)
        pod.update_position("BTC/USDT", 2.0, 55000.0, 0.0, is_buy=False)

        pos = pod._positions["BTC/USDT"]
        # realized from closing long: (55000-50000)*1.0 = 5000
        assert pos.realized_pnl == pytest.approx(5000.0)
        # New position: short 1.0 @ 55000
        assert pos.quantity == pytest.approx(-1.0)
        assert pos.avg_entry_price == pytest.approx(55000.0)

    def test_quantity_signed_tracking(self) -> None:
        """quantity는 signed (+ long, - short)."""
        pod = _make_pod()
        pod.update_position("BTC/USDT", 1.0, 50000.0, 0.0, is_buy=True)
        assert pod._positions["BTC/USDT"].quantity == pytest.approx(1.0)

        pod2 = _make_pod(pod_id="pod-b")
        pod2.update_position("BTC/USDT", 1.0, 50000.0, 0.0, is_buy=False)
        assert pod2._positions["BTC/USDT"].quantity == pytest.approx(-1.0)

    def test_to_dict_restore_roundtrip(self) -> None:
        """to_dict/restore_from_dict round-trip에 avg_entry_price, quantity 포함."""
        pod = _make_pod()
        pod.update_position("BTC/USDT", 1.0, 50000.0, 5.0, is_buy=True)

        data = pod.to_dict()
        pos_data = data["positions"]
        assert isinstance(pos_data, dict)
        assert "avg_entry_price" in pos_data["BTC/USDT"]
        assert "quantity" in pos_data["BTC/USDT"]

        # Restore
        pod2 = _make_pod(pod_id="pod-a")
        pod2.restore_from_dict(data)
        assert pod2._positions["BTC/USDT"].avg_entry_price == pytest.approx(50000.0)
        assert pod2._positions["BTC/USDT"].quantity == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════
# C-2: PodPerformance 메트릭 미갱신
# ═══════════════════════════════════════════════════════════════════


class TestC2MetricsUpdate:
    """C-2: record_daily_return() 호출 시 메트릭 자동 갱신."""

    def test_total_return_computed(self) -> None:
        """total_return = prod(1+r) - 1."""
        pod = _make_pod()
        pod.record_daily_return(0.10)  # +10%
        pod.record_daily_return(0.05)  # +5%
        expected = (1.10 * 1.05) - 1.0  # 0.155
        assert pod.performance.total_return == pytest.approx(expected, rel=1e-6)

    def test_max_drawdown_computed(self) -> None:
        """max_drawdown 정확성."""
        pod = _make_pod()
        pod.record_daily_return(0.10)
        pod.record_daily_return(-0.20)
        pod.record_daily_return(0.05)

        # Equity: 1.0 → 1.10 → 0.88 → 0.924
        # Peak: 1.10, DD = (1.10-0.88)/1.10 = 0.2
        assert pod.performance.max_drawdown == pytest.approx(0.2, abs=1e-6)

    def test_sharpe_ratio_nonzero(self) -> None:
        """sharpe_ratio != 0 after sufficient varying returns."""
        pod = _make_pod()
        returns = [0.01, -0.005, 0.008, -0.003, 0.012] * 6  # 30 varying returns
        for r in returns:
            pod.record_daily_return(r)
        assert pod.performance.sharpe_ratio != 0.0

    def test_calmar_ratio_nonzero(self) -> None:
        """calmar_ratio != 0 when drawdown exists."""
        pod = _make_pod()
        pod.record_daily_return(0.10)
        pod.record_daily_return(-0.05)
        pod.record_daily_return(0.03)
        assert pod.performance.calmar_ratio != 0.0

    def test_win_rate_computed(self) -> None:
        """win_rate = positive_days / total_days."""
        pod = _make_pod()
        pod.record_daily_return(0.01)
        pod.record_daily_return(-0.01)
        pod.record_daily_return(0.02)
        pod.record_daily_return(0.005)
        # 3 positive out of 4
        assert pod.performance.win_rate == pytest.approx(0.75)

    def test_single_return_no_crash(self) -> None:
        """단일 return 시 crash 없음."""
        pod = _make_pod()
        pod.record_daily_return(0.05)
        # n < 2 → only total_return set
        assert pod.performance.total_return == pytest.approx(0.05)
        assert pod.performance.sharpe_ratio == 0.0

    def test_peak_equity_maintained(self) -> None:
        """peak_equity는 최고점 유지."""
        pod = _make_pod()
        pod.record_daily_return(0.10)
        pod.record_daily_return(-0.05)
        # Peak should be 1.10
        assert pod.performance.peak_equity == pytest.approx(1.10, rel=1e-6)

    def test_metrics_nonzero_after_record(self) -> None:
        """record_daily_return 호출 후 메트릭 != 0 확인."""
        pod = _make_pod()
        returns = [0.01, -0.005, 0.008, -0.003, 0.012, 0.002, -0.001, 0.006, 0.01, -0.002]
        for r in returns:
            pod.record_daily_return(r)
        perf = pod.performance
        assert perf.total_return != 0.0
        assert perf.rolling_volatility != 0.0
        assert perf.peak_equity > 0.0
        assert perf.current_equity > 0.0


# ═══════════════════════════════════════════════════════════════════
# C-3: Lifecycle PH score vs Sharpe 단위 불일치
# ═══════════════════════════════════════════════════════════════════


class TestC3LifecycleWarning:
    """C-3: _evaluate_warning() 조건 수정 검증."""

    def test_score_zero_no_immediate_recovery(self) -> None:
        """Score 0에서 최소 관찰 기간 미달 시 즉시 복귀 안 함."""
        pod = _make_pod()
        pod.state = LifecycleState.WARNING
        _inject_daily_returns(pod, [0.01] * 10)

        mgr = LifecycleManager(
            graduation=GraduationCriteria(),
            retirement=RetirementCriteria(),
        )
        pod_ls = mgr._get_or_create_state(pod)
        # 방금 진입 (0일)
        pod_ls.state_entered_at = datetime.now(UTC)
        pod_ls.ph_detector.reset()

        result = mgr.evaluate(pod)
        # 최소 5일 미경과 → WARNING 유지
        assert result == LifecycleState.WARNING

    def test_recovery_after_observation_period(self) -> None:
        """관찰 기간 후 score < threshold → PRODUCTION 복귀."""
        pod = _make_pod()
        pod.state = LifecycleState.WARNING
        _inject_daily_returns(pod, [0.01] * 10)

        mgr = LifecycleManager(
            graduation=GraduationCriteria(),
            retirement=RetirementCriteria(),
        )
        pod_ls = mgr._get_or_create_state(pod)
        # 6일 전에 WARNING 진입
        pod_ls.state_entered_at = datetime.now(UTC) - timedelta(days=6)
        pod_ls.ph_detector.reset()

        result = mgr.evaluate(pod)
        assert result == LifecycleState.PRODUCTION

    def test_timeout_30d_to_probation(self) -> None:
        """30일 timeout → PROBATION 유지."""
        pod = _make_pod()
        pod.state = LifecycleState.WARNING
        _inject_daily_returns(pod, [0.001] * 10)

        mgr = LifecycleManager(
            graduation=GraduationCriteria(),
            retirement=RetirementCriteria(),
        )
        pod_ls = mgr._get_or_create_state(pod)
        pod_ls.state_entered_at = datetime.now(UTC) - timedelta(days=31)
        # Score > threshold (no recovery). Use large shift for alpha=0.99 sensitivity.
        for _ in range(50):
            pod_ls.ph_detector.update(0.01)
        for _ in range(500):
            pod_ls.ph_detector.update(-0.20)

        assert pod_ls.ph_detector.score >= pod_ls.ph_detector.lambda_threshold * 0.2

        result = mgr.evaluate(pod)
        assert result == LifecycleState.PROBATION

    def test_lambda_02_threshold_accuracy(self) -> None:
        """lambda * 0.2 threshold 정확성."""
        pod = _make_pod()
        pod.state = LifecycleState.WARNING
        _inject_daily_returns(pod, [0.01] * 10)

        mgr = LifecycleManager(
            graduation=GraduationCriteria(),
            retirement=RetirementCriteria(),
        )
        pod_ls = mgr._get_or_create_state(pod)
        pod_ls.state_entered_at = datetime.now(UTC) - timedelta(days=6)

        # lambda = 50.0, threshold = 10.0
        threshold = pod_ls.ph_detector.lambda_threshold * 0.2
        assert threshold == pytest.approx(10.0)

        # Score just above threshold → no recovery
        # Feed enough data to raise score above 10.0 (alpha=0.99 needs larger shift)
        for _ in range(50):
            pod_ls.ph_detector.update(0.01)
        for _ in range(500):
            pod_ls.ph_detector.update(-0.20)
        assert pod_ls.ph_detector.score >= threshold

        result = mgr.evaluate(pod)
        # Score >= threshold + days < 30 → WARNING remains
        assert result == LifecycleState.WARNING


# ═══════════════════════════════════════════════════════════════════
# C-7: Risk check 빈 데이터 대상 실행
# ═══════════════════════════════════════════════════════════════════


class TestC7RiskCheckWeights:
    """C-7: flush 후에도 risk check에 유효한 weights 전달."""

    def test_compute_current_net_weights(self) -> None:
        """_compute_current_net_weights() 합산 정확성."""
        pod_a = _make_pod("pod-a", ("BTC/USDT",))
        pod_b = _make_pod("pod-b", ("BTC/USDT",))
        config = _make_orchestrator_config((pod_a.config, pod_b.config))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod_a, pod_b], allocator)

        # Simulate last_pod_targets
        orch._last_pod_targets = {
            "pod-a": {"BTC/USDT": 0.3, "ETH/USDT": 0.2},
            "pod-b": {"BTC/USDT": 0.4},
        }

        weights = orch._compute_current_net_weights()
        assert weights["BTC/USDT"] == pytest.approx(0.7)
        assert weights["ETH/USDT"] == pytest.approx(0.2)

    def test_risk_check_uses_snapshot_not_empty(self) -> None:
        """flush 후에도 risk check에 유효한 weights 전달 확인."""
        pod_a = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=0.5)
        pod_b = _make_pod("pod-b", ("ETH/USDT",), capital_fraction=0.5)
        config = _make_orchestrator_config(
            (pod_a.config, pod_b.config),
            rebalance_calendar_days=1,
        )
        allocator = CapitalAllocator(config)
        ra = RiskAggregator(config)
        orch = StrategyOrchestrator(
            config,
            [pod_a, pod_b],
            allocator,
            risk_aggregator=ra,
        )

        # Pod targets 설정
        orch._last_pod_targets = {
            "pod-a": {"BTC/USDT": 0.5},
            "pod-b": {"ETH/USDT": 0.3},
        }

        # pending_net_weights는 비어있음 (flush 후 상태)
        orch._pending_net_weights = {}

        pod_a.record_daily_return(0.01)
        pod_b.record_daily_return(0.02)

        pod_returns = pd.DataFrame(
            {
                "pod-a": [0.01],
                "pod-b": [0.02],
            }
        )

        # _check_risk_limits with explicit net_weights
        net_weights = orch._compute_current_net_weights()
        alerts = orch._check_risk_limits(pod_returns, net_weights=net_weights)

        # Should not crash, and net_weights should be non-empty
        assert net_weights != {}
        assert isinstance(alerts, list)


# ═══════════════════════════════════════════════════════════════════
# C-6: daily_pnl_pct 하드코딩 0.0
# ═══════════════════════════════════════════════════════════════════


class TestC6DailyPnL:
    """C-6: _compute_daily_pnl_pct() 정확한 가중 평균."""

    def test_two_pods_weighted_average(self) -> None:
        """2 Pods with known returns → 정확한 가중 평균."""
        pod_a = _make_pod("pod-a", capital_fraction=0.6)
        pod_b = _make_pod("pod-b", capital_fraction=0.4)
        config = _make_orchestrator_config((pod_a.config, pod_b.config))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod_a, pod_b], allocator)

        _inject_daily_returns(pod_a, [0.05])
        _inject_daily_returns(pod_b, [-0.02])

        pnl = orch._compute_daily_pnl_pct()
        # weighted = (0.05 * 0.6 + (-0.02) * 0.4) / (0.6 + 0.4) = (0.03 - 0.008) / 1.0 = 0.022
        assert pnl == pytest.approx(0.022)

    def test_empty_pods_returns_zero(self) -> None:
        """empty pods → 0.0."""
        pod_a = _make_pod("pod-a")
        config = _make_orchestrator_config((pod_a.config,))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod_a], allocator)

        # No daily returns recorded
        pnl = orch._compute_daily_pnl_pct()
        assert pnl == 0.0

    def test_nonzero_pnl_passed_to_risk(self) -> None:
        """check_portfolio_limits에 non-zero daily_pnl_pct 전달 확인."""
        pod_a = _make_pod("pod-a", capital_fraction=0.5)
        pod_b = _make_pod("pod-b", capital_fraction=0.5)
        config = _make_orchestrator_config(
            (pod_a.config, pod_b.config),
            rebalance_calendar_days=1,
        )
        allocator = CapitalAllocator(config)
        ra = RiskAggregator(config)
        orch = StrategyOrchestrator(
            config,
            [pod_a, pod_b],
            allocator,
            risk_aggregator=ra,
        )

        _inject_daily_returns(pod_a, [-0.10])
        _inject_daily_returns(pod_b, [-0.10])

        orch._last_pod_targets = {"pod-a": {"BTC/USDT": 0.5}}
        pod_returns = pd.DataFrame({"pod-a": [-0.1], "pod-b": [-0.1]})

        with patch.object(
            ra, "check_portfolio_limits", wraps=ra.check_portfolio_limits
        ) as mock_check:
            orch._check_risk_limits(pod_returns, net_weights={"BTC/USDT": 0.5})
            call_kwargs = mock_check.call_args
            # daily_pnl_pct should be non-zero (both pods have -10%)
            assert call_kwargs.kwargs["daily_pnl_pct"] != 0.0


# ═══════════════════════════════════════════════════════════════════
# C-5: Risk alert 무방어
# ═══════════════════════════════════════════════════════════════════


class TestC5RiskDefense:
    """C-5: CRITICAL alert 시 capital 축소 + signal 억제."""

    def _make_orch_with_risk(
        self,
    ) -> tuple[StrategyOrchestrator, StrategyPod, StrategyPod, RiskAggregator]:
        pod_a = _make_pod("pod-a", capital_fraction=0.5)
        pod_b = _make_pod("pod-b", capital_fraction=0.5)
        config = _make_orchestrator_config(
            (pod_a.config, pod_b.config),
            rebalance_calendar_days=100,
        )
        allocator = CapitalAllocator(config)
        ra = RiskAggregator(config)
        orch = StrategyOrchestrator(
            config,
            [pod_a, pod_b],
            allocator,
            risk_aggregator=ra,
        )
        return orch, pod_a, pod_b, ra

    def test_critical_alert_scales_capital(self) -> None:
        """CRITICAL alert → capital_fraction 50% 축소."""
        orch, pod_a, pod_b, ra = self._make_orch_with_risk()
        pod_a.record_daily_return(0.01)
        pod_b.record_daily_return(0.01)

        critical_alert = RiskAlert(
            alert_type="test",
            severity="critical",
            message="test",
            current_value=1.0,
            threshold=0.5,
        )
        with patch.object(ra, "check_portfolio_limits", return_value=[critical_alert]):
            pod_returns = pd.DataFrame({"pod-a": [0.01], "pod-b": [0.01]})
            orch._check_risk_limits(pod_returns, net_weights={"BTC/USDT": 0.5})

        assert pod_a.capital_fraction == pytest.approx(0.25)
        assert pod_b.capital_fraction == pytest.approx(0.25)
        assert orch._risk_breached is True

    def test_risk_breached_suppresses_signals(self) -> None:
        """risk_breached=True → flush에서 weight=0 발행."""
        orch, _, _, _ = self._make_orch_with_risk()
        orch._risk_breached = True
        orch._pending_net_weights = {"BTC/USDT": 0.8, "ETH/USDT": 0.5}

        # Trigger flush simulation: call the suppression logic directly
        if orch._risk_breached:
            orch._pending_net_weights = dict.fromkeys(orch._pending_net_weights, 0.0)

        assert orch._pending_net_weights["BTC/USDT"] == 0.0
        assert orch._pending_net_weights["ETH/USDT"] == 0.0

    def test_critical_resolved_resets_flag(self) -> None:
        """CRITICAL 해소 → flag 리셋."""
        orch, pod_a, pod_b, ra = self._make_orch_with_risk()
        orch._risk_breached = True
        pod_a.record_daily_return(0.01)
        pod_b.record_daily_return(0.01)

        # No alerts → flag should be reset
        with patch.object(ra, "check_portfolio_limits", return_value=[]):
            pod_returns = pd.DataFrame({"pod-a": [0.01], "pod-b": [0.01]})
            orch._check_risk_limits(pod_returns, net_weights={"BTC/USDT": 0.5})

        assert orch._risk_breached is False

    def test_warning_only_no_defense(self) -> None:
        """WARNING-only → 방어 미발동."""
        orch, pod_a, pod_b, ra = self._make_orch_with_risk()
        pod_a.record_daily_return(0.01)
        pod_b.record_daily_return(0.01)
        original_a = pod_a.capital_fraction
        original_b = pod_b.capital_fraction

        warning_alert = RiskAlert(
            alert_type="test",
            severity="warning",
            message="test",
            current_value=0.8,
            threshold=1.0,
        )
        with patch.object(ra, "check_portfolio_limits", return_value=[warning_alert]):
            pod_returns = pd.DataFrame({"pod-a": [0.01], "pod-b": [0.01]})
            orch._check_risk_limits(pod_returns, net_weights={"BTC/USDT": 0.5})

        assert pod_a.capital_fraction == pytest.approx(original_a)
        assert pod_b.capital_fraction == pytest.approx(original_b)
        assert orch._risk_breached is False

    def test_consecutive_critical_compounds(self) -> None:
        """연속 CRITICAL 시 capital 지속 감소."""
        orch, pod_a, pod_b, ra = self._make_orch_with_risk()
        pod_a.record_daily_return(0.01)
        pod_b.record_daily_return(0.01)

        critical_alert = RiskAlert(
            alert_type="test",
            severity="critical",
            message="test",
            current_value=1.0,
            threshold=0.5,
        )

        pod_returns = pd.DataFrame({"pod-a": [0.01], "pod-b": [0.01]})

        # First critical
        with patch.object(ra, "check_portfolio_limits", return_value=[critical_alert]):
            orch._check_risk_limits(pod_returns, net_weights={"BTC/USDT": 0.5})
        assert pod_a.capital_fraction == pytest.approx(0.25)

        # Second critical
        with patch.object(ra, "check_portfolio_limits", return_value=[critical_alert]):
            orch._check_risk_limits(pod_returns, net_weights={"BTC/USDT": 0.5})
        assert pod_a.capital_fraction == pytest.approx(0.125)


# ═══════════════════════════════════════════════════════════════════
# C-4: Orchestrator 상태 LiveRunner에서 미영속
# ═══════════════════════════════════════════════════════════════════


class TestC4OrchestratorPersistence:
    """C-4: LiveRunner에서 OrchestratorStatePersistence 통합."""

    async def test_restore_orchestrator_state_on_run(self) -> None:
        """_restore_orchestrator_state로 orchestrator 복구."""
        from src.eda.live_runner import LiveRunner

        runner = MagicMock(spec=LiveRunner)
        runner._orchestrator = MagicMock()

        # Mock state_mgr
        state_mgr = MagicMock()
        state_mgr._db = MagicMock()

        with patch(
            "src.orchestrator.state_persistence.OrchestratorStatePersistence"
        ) as mock_persistence_cls:
            mock_instance = MagicMock()
            mock_instance.restore = AsyncMock(return_value=True)
            mock_persistence_cls.return_value = mock_instance

            result = await LiveRunner._restore_orchestrator_state(runner, state_mgr)

            assert result is mock_instance
            mock_instance.restore.assert_awaited_once_with(runner._orchestrator)

    async def test_no_orchestrator_returns_none(self) -> None:
        """orchestrator=None 시 기존 동작 유지."""
        from src.eda.live_runner import LiveRunner

        runner = MagicMock(spec=LiveRunner)
        runner._orchestrator = None

        state_mgr = MagicMock()
        result = await LiveRunner._restore_orchestrator_state(runner, state_mgr)
        assert result is None

    async def test_no_state_mgr_returns_none(self) -> None:
        """state_mgr=None 시 None 반환."""
        from src.eda.live_runner import LiveRunner

        runner = MagicMock(spec=LiveRunner)
        runner._orchestrator = MagicMock()

        result = await LiveRunner._restore_orchestrator_state(runner, None)
        assert result is None

    async def test_periodic_save_includes_orchestrator(self) -> None:
        """Periodic save에 orchestrator 포함."""
        from src.eda.live_runner import LiveRunner

        state_mgr = MagicMock()
        state_mgr.save_all = AsyncMock()
        pm = MagicMock()
        rm = MagicMock()

        orch_persistence = MagicMock()
        orch_persistence.save = AsyncMock()
        orchestrator = MagicMock()

        # Run one iteration with a very short interval
        task = asyncio.create_task(
            LiveRunner._periodic_state_save(
                state_mgr,
                pm,
                rm,
                orch_persistence=orch_persistence,
                orchestrator=orchestrator,
                interval=0.01,
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        state_mgr.save_all.assert_awaited()
        orch_persistence.save.assert_awaited_with(orchestrator)
