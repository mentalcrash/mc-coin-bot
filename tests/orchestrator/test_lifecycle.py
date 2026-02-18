"""Tests for LifecycleManager — Pod 생애주기 자동 관리."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from src.orchestrator.config import GraduationCriteria, PodConfig, RetirementCriteria
from src.orchestrator.lifecycle import LifecycleManager
from src.orchestrator.models import LifecycleState
from src.orchestrator.pod import StrategyPod
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


def _make_pod(
    pod_id: str = "pod-a",
    symbols: tuple[str, ...] = ("BTC/USDT",),
    capital_fraction: float = 0.25,
) -> StrategyPod:
    """테스트용 StrategyPod 생성."""
    config = PodConfig(
        pod_id=pod_id,
        strategy_name="tsmom",
        symbols=symbols,
        initial_fraction=capital_fraction,
        max_fraction=0.40,
        min_fraction=0.02,
    )
    strategy = SimpleTestStrategy()
    pod = StrategyPod(config=config, strategy=strategy, capital_fraction=capital_fraction)
    pod._warmup = 3
    return pod


def _set_pod_performance(pod: StrategyPod, **kwargs: object) -> None:
    """PodPerformance 필드를 직접 설정."""
    perf = pod.performance
    for key, value in kwargs.items():
        setattr(perf, key, value)


def _inject_daily_returns(pod: StrategyPod, returns: list[float]) -> None:
    """Pod의 _daily_returns를 직접 주입 (테스트 전용)."""
    pod._daily_returns = list(returns)  # type: ignore[attr-defined]
    pod._performance.live_days = len(returns)  # type: ignore[attr-defined]


def _make_manager(
    graduation: GraduationCriteria | None = None,
    retirement: RetirementCriteria | None = None,
) -> LifecycleManager:
    """테스트용 LifecycleManager 생성."""
    grad = graduation or GraduationCriteria()
    ret = retirement or RetirementCriteria()
    return LifecycleManager(graduation=grad, retirement=ret)


def _graduation_ready_pod(pod_id: str = "pod-a") -> StrategyPod:
    """졸업 기준을 모두 충족하는 Pod 생성."""
    pod = _make_pod(pod_id=pod_id)
    _set_pod_performance(
        pod,
        live_days=100,
        sharpe_ratio=1.5,
        max_drawdown=0.10,
        trade_count=50,
        calmar_ratio=1.2,
    )
    _inject_daily_returns(pod, [0.001] * 100)
    return pod


# ── TestHardStops ─────────────────────────────────────────────────


class TestHardStops:
    def test_mdd_breach_from_incubation_retires(self) -> None:
        """INCUBATION + MDD >= 25% → RETIRED."""
        pod = _make_pod()
        _set_pod_performance(pod, max_drawdown=0.25)

        mgr = _make_manager()
        result = mgr.evaluate(pod)
        assert result == LifecycleState.RETIRED
        assert pod.state == LifecycleState.RETIRED

    def test_mdd_breach_from_production_retires(self) -> None:
        """PRODUCTION + MDD >= 25% → RETIRED."""
        pod = _make_pod()
        pod.state = LifecycleState.PRODUCTION
        _set_pod_performance(pod, max_drawdown=0.26)

        mgr = _make_manager()
        result = mgr.evaluate(pod)
        assert result == LifecycleState.RETIRED

    def test_mdd_breach_from_warning_retires(self) -> None:
        """WARNING + MDD >= 25% → RETIRED."""
        pod = _make_pod()
        pod.state = LifecycleState.WARNING
        _set_pod_performance(pod, max_drawdown=0.30)

        mgr = _make_manager()
        result = mgr.evaluate(pod)
        assert result == LifecycleState.RETIRED

    def test_consecutive_loss_months_retires(self) -> None:
        """6개월 연속 손실 → RETIRED."""
        pod = _make_pod()
        pod.state = LifecycleState.PRODUCTION

        # 6개월 * 30일 = 180일의 음수 수익률
        returns = [-0.001] * 180
        _inject_daily_returns(pod, returns)

        mgr = _make_manager()
        result = mgr.evaluate(pod)
        assert result == LifecycleState.RETIRED

    def test_mdd_boundary_at_25_percent(self) -> None:
        """MDD = 25% → RETIRED (경계값)."""
        pod = _make_pod()
        _set_pod_performance(pod, max_drawdown=0.25)

        mgr = _make_manager()
        result = mgr.evaluate(pod)
        assert result == LifecycleState.RETIRED

    def test_mdd_just_below_25_percent_survives(self) -> None:
        """MDD = 24.9% → 퇴출 안됨."""
        pod = _make_pod()
        _set_pod_performance(pod, max_drawdown=0.249)

        mgr = _make_manager()
        result = mgr.evaluate(pod)
        assert result != LifecycleState.RETIRED


# ── TestGraduation ───────────────────────────────────────────────


class TestGraduation:
    def test_all_criteria_met_graduates(self) -> None:
        """모든 졸업 기준 충족 → PRODUCTION."""
        pod = _graduation_ready_pod()
        mgr = _make_manager()
        result = mgr.evaluate(pod)
        assert result == LifecycleState.PRODUCTION

    def test_insufficient_live_days_stays(self) -> None:
        """live_days 미달 → INCUBATION 유지."""
        pod = _graduation_ready_pod()
        _set_pod_performance(pod, live_days=20)

        mgr = _make_manager()
        result = mgr.evaluate(pod)
        assert result == LifecycleState.INCUBATION

    def test_low_sharpe_stays(self) -> None:
        """sharpe 미달 → INCUBATION 유지."""
        pod = _graduation_ready_pod()
        _set_pod_performance(pod, sharpe_ratio=0.3)

        mgr = _make_manager()
        result = mgr.evaluate(pod)
        assert result == LifecycleState.INCUBATION

    def test_high_mdd_stays(self) -> None:
        """MDD 초과 → INCUBATION 유지."""
        pod = _graduation_ready_pod()
        _set_pod_performance(pod, max_drawdown=0.22)

        mgr = _make_manager()
        result = mgr.evaluate(pod)
        assert result == LifecycleState.INCUBATION

    def test_only_from_incubation(self) -> None:
        """PRODUCTION에서는 졸업 체크 안함."""
        pod = _graduation_ready_pod()
        pod.state = LifecycleState.PRODUCTION
        _set_pod_performance(pod, max_drawdown=0.10)

        mgr = _make_manager()
        result = mgr.evaluate(pod)
        # PRODUCTION에서는 degradation 체크, graduation 아님
        assert result == LifecycleState.PRODUCTION

    def test_high_correlation_blocks(self) -> None:
        """포트폴리오 상관계수 > 0.5 → 졸업 불가."""
        pod = _graduation_ready_pod()
        # Pod과 동일한 수익률 → 상관계수 = 1.0
        portfolio_returns = pd.Series(list(pod.daily_returns), dtype=float)

        mgr = _make_manager()
        result = mgr.evaluate(pod, portfolio_returns)
        assert result == LifecycleState.INCUBATION

    def test_no_portfolio_returns_skips_correlation(self) -> None:
        """portfolio_returns=None → 상관계수 체크 skip."""
        pod = _graduation_ready_pod()

        mgr = _make_manager()
        result = mgr.evaluate(pod, portfolio_returns=None)
        assert result == LifecycleState.PRODUCTION


# ── TestDegradation ──────────────────────────────────────────────


class TestDegradation:
    def test_ph_triggers_warning(self) -> None:
        """PH 감지 → PRODUCTION → WARNING."""
        pod = _make_pod()
        pod.state = LifecycleState.PRODUCTION

        # Prevent consecutive_loss_months hard stop from interfering
        retirement = RetirementCriteria(consecutive_loss_months=100)
        mgr = _make_manager(retirement=retirement)

        # alpha=0.99에서 lambda=50 도달을 위해 큰 시프트 필요
        normal_returns = [0.01] * 100
        bad_returns = [-1.0] * 1500

        all_returns = normal_returns + bad_returns
        detected = False
        for i in range(len(all_returns)):
            _inject_daily_returns(pod, all_returns[: i + 1])
            if pod.state == LifecycleState.RETIRED:
                break
            mgr.evaluate(pod)
            if pod.state == LifecycleState.WARNING:
                detected = True
                break

        assert detected, "PH detector should trigger WARNING"

    def test_stable_stays_production(self) -> None:
        """안정적 수익 → PRODUCTION 유지."""
        pod = _make_pod()
        pod.state = LifecycleState.PRODUCTION
        _inject_daily_returns(pod, [0.005] * 50)

        mgr = _make_manager()
        # 여러 번 evaluate (각 호출에서 latest return만 PH에 공급)
        for _ in range(50):
            mgr.evaluate(pod)

        assert pod.state == LifecycleState.PRODUCTION


# ── TestWarningRecovery ──────────────────────────────────────────


class TestWarningRecovery:
    def test_recovery_within_30d_returns_production(self) -> None:
        """WARNING에서 PH score < lambda*0.2 AND 5일 경과 → PRODUCTION 복귀."""
        pod = _make_pod()
        pod.state = LifecycleState.WARNING
        _inject_daily_returns(pod, [0.01] * 10)

        mgr = _make_manager()
        pod_ls = mgr._get_or_create_state(pod)
        # 5일 전에 WARNING 진입 (최소 관찰 기간 충족)
        pod_ls.state_entered_at = datetime.now(UTC) - timedelta(days=6)
        # PH score가 낮으면 recovery
        pod_ls.ph_detector.reset()

        result = mgr.evaluate(pod)
        assert result == LifecycleState.PRODUCTION

    def test_no_recovery_30d_escalates_probation(self) -> None:
        """WARNING 30일 경과, 미회복 → PROBATION."""
        pod = _make_pod()
        pod.state = LifecycleState.WARNING
        _inject_daily_returns(pod, [0.001] * 10)

        mgr = _make_manager()
        pod_ls = mgr._get_or_create_state(pod)
        # 30일 전으로 설정
        pod_ls.state_entered_at = datetime.now(UTC) - timedelta(days=31)

        # PH score를 높게 유지 (정상→악화 shift로 score 누적)
        # score > lambda * 0.2 = 10.0이 되어야 recovery 불가
        # alpha=0.99에서는 큰 shift 필요 (x_mean 수렴이 빠름)
        for _ in range(50):
            pod_ls.ph_detector.update(0.01)
        for _ in range(500):
            pod_ls.ph_detector.update(-0.20)

        assert pod_ls.ph_detector.score >= pod_ls.ph_detector.lambda_threshold * 0.2

        result = mgr.evaluate(pod)
        assert result == LifecycleState.PROBATION


# ── TestProbation ────────────────────────────────────────────────


class TestProbation:
    def test_strong_recovery_returns_production(self) -> None:
        """PROBATION에서 sharpe >= 1.0 AND ph_score <= 0 → PRODUCTION."""
        pod = _make_pod()
        pod.state = LifecycleState.PROBATION
        _set_pod_performance(pod, sharpe_ratio=1.5)
        _inject_daily_returns(pod, [0.01] * 10)

        mgr = _make_manager()
        pod_ls = mgr._get_or_create_state(pod)
        pod_ls.ph_detector.reset()  # score = 0

        result = mgr.evaluate(pod)
        assert result == LifecycleState.PRODUCTION

    def test_expired_probation_retires(self) -> None:
        """PROBATION 30일(default) 경과, 미회복 → RETIRED."""
        pod = _make_pod()
        pod.state = LifecycleState.PROBATION
        _set_pod_performance(pod, sharpe_ratio=0.3)  # 회복 안됨
        _inject_daily_returns(pod, [0.001] * 10)

        mgr = _make_manager()
        pod_ls = mgr._get_or_create_state(pod)
        pod_ls.state_entered_at = datetime.now(UTC) - timedelta(days=31)
        # ph_score > 0 (회복 안됨)
        for _ in range(10):
            pod_ls.ph_detector.update(-0.05)

        result = mgr.evaluate(pod)
        assert result == LifecycleState.RETIRED


# ── TestRetired ──────────────────────────────────────────────────


class TestRetired:
    def test_retired_is_terminal(self) -> None:
        """RETIRED 상태는 terminal — 더 이상 전이 없음."""
        pod = _make_pod()
        pod.state = LifecycleState.RETIRED

        mgr = _make_manager()
        result = mgr.evaluate(pod)
        assert result == LifecycleState.RETIRED

    def test_retired_with_good_performance_stays_retired(self) -> None:
        """RETIRED + 좋은 성과여도 복귀 불가."""
        pod = _make_pod()
        pod.state = LifecycleState.RETIRED
        _set_pod_performance(
            pod,
            live_days=200,
            sharpe_ratio=3.0,
            max_drawdown=0.05,
            trade_count=100,
            calmar_ratio=2.0,
        )

        mgr = _make_manager()
        result = mgr.evaluate(pod)
        assert result == LifecycleState.RETIRED


# ── TestIntegration ──────────────────────────────────────────────


class TestIntegration:
    def test_full_lifecycle_incubation_to_production(self) -> None:
        """INCUBATION → PRODUCTION 전체 흐름."""
        pod = _graduation_ready_pod()
        mgr = _make_manager()

        assert pod.state == LifecycleState.INCUBATION
        mgr.evaluate(pod)
        assert pod.state == LifecycleState.PRODUCTION

    def test_multiple_pods_independent(self) -> None:
        """여러 Pod이 독립적으로 평가됨."""
        pod_a = _graduation_ready_pod(pod_id="pod-a")
        pod_b = _make_pod(pod_id="pod-b")
        _set_pod_performance(pod_b, max_drawdown=0.30)

        mgr = _make_manager()
        mgr.evaluate(pod_a)
        mgr.evaluate(pod_b)

        assert pod_a.state == LifecycleState.PRODUCTION
        assert pod_b.state == LifecycleState.RETIRED

    def test_detector_reset_on_recovery(self) -> None:
        """WARNING → PRODUCTION 복귀 시 PH detector reset."""
        pod = _make_pod()
        pod.state = LifecycleState.WARNING
        _inject_daily_returns(pod, [0.01] * 10)

        mgr = _make_manager()
        pod_ls = mgr._get_or_create_state(pod)
        # 5일 전에 WARNING 진입 (최소 관찰 기간 충족)
        pod_ls.state_entered_at = datetime.now(UTC) - timedelta(days=6)
        # Feed some data before recovery
        for _ in range(5):
            pod_ls.ph_detector.update(0.01)
        assert pod_ls.ph_detector.n_observations > 0

        # Trigger recovery (low ph_score)
        pod_ls.ph_detector.reset()
        mgr.evaluate(pod)

        assert pod.state == LifecycleState.PRODUCTION
        # detector should be reset after transition
        assert pod_ls.ph_detector.n_observations == 0

    def test_consecutive_loss_months_reset_on_profit(self) -> None:
        """수익 달 발생 시 연속 손실 카운트 리셋."""
        pod = _make_pod()
        pod.state = LifecycleState.PRODUCTION

        # 2개월 손실 + 1개월 수익
        returns = [-0.001] * 60 + [0.01] * 30
        _inject_daily_returns(pod, returns)

        mgr = _make_manager()
        mgr.evaluate(pod)

        pod_ls = mgr._pod_states[pod.pod_id]
        assert pod_ls.consecutive_loss_months == 0

    def test_bar_timestamp_warning_to_probation(self) -> None:
        """H-3: bar_timestamp 전달 시 WARNING→PROBATION timeout 정상 동작."""
        pod = _make_pod()
        pod.state = LifecycleState.WARNING
        _inject_daily_returns(pod, [0.001] * 10)

        mgr = _make_manager()
        pod_ls = mgr._get_or_create_state(pod)
        # WARNING 진입을 2024-01-01로 설정
        warning_entered = datetime(2024, 1, 1, tzinfo=UTC)
        pod_ls.state_entered_at = warning_entered

        # PH score를 높게 유지 (recovery 불가, alpha=0.99 needs larger shift)
        for _ in range(50):
            pod_ls.ph_detector.update(0.01)
        for _ in range(500):
            pod_ls.ph_detector.update(-0.20)

        assert pod_ls.ph_detector.score >= pod_ls.ph_detector.lambda_threshold * 0.2

        # bar_timestamp = 2024-02-01 (31일 후) → timeout
        bar_ts = datetime(2024, 2, 1, tzinfo=UTC)
        result = mgr.evaluate(pod, bar_timestamp=bar_ts)
        assert result == LifecycleState.PROBATION

    def test_bar_timestamp_probation_to_retired(self) -> None:
        """H-3: bar_timestamp 전달 시 PROBATION→RETIRED timeout 정상 동작."""
        pod = _make_pod()
        pod.state = LifecycleState.PROBATION
        _set_pod_performance(pod, sharpe_ratio=0.3)
        _inject_daily_returns(pod, [0.001] * 10)

        mgr = _make_manager()
        pod_ls = mgr._get_or_create_state(pod)
        probation_entered = datetime(2024, 3, 1, tzinfo=UTC)
        pod_ls.state_entered_at = probation_entered

        # ph_score > 0 (회복 안됨)
        for _ in range(10):
            pod_ls.ph_detector.update(-0.05)

        # bar_timestamp = 2024-04-01 (31일 후) → expired
        bar_ts = datetime(2024, 4, 1, tzinfo=UTC)
        result = mgr.evaluate(pod, bar_timestamp=bar_ts)
        assert result == LifecycleState.RETIRED

    def test_bar_timestamp_sets_state_entered_at(self) -> None:
        """H-3: _transition 시 state_entered_at = bar_timestamp."""
        pod = _graduation_ready_pod()
        bar_ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)

        mgr = _make_manager()
        result = mgr.evaluate(pod, bar_timestamp=bar_ts)
        assert result == LifecycleState.PRODUCTION

        pod_ls = mgr._pod_states[pod.pod_id]
        assert pod_ls.state_entered_at == bar_ts

    def test_bar_timestamp_none_uses_wall_clock(self) -> None:
        """H-3: bar_timestamp=None → wall clock fallback."""
        pod = _graduation_ready_pod()
        before = datetime.now(UTC)

        mgr = _make_manager()
        mgr.evaluate(pod, bar_timestamp=None)

        pod_ls = mgr._pod_states[pod.pod_id]
        after = datetime.now(UTC)
        assert before <= pod_ls.state_entered_at <= after

    def test_geometric_monthly_return_detects_loss(self) -> None:
        """H-5: 산술합 ≈ 0이지만 기하 복리 < 0인 케이스에서 loss 감지."""
        pod = _make_pod()
        pod.state = LifecycleState.PRODUCTION

        # [-0.10, +0.10] * 15 = 30일
        # 산술합 = 0.0, 기하 복리 = (0.9 * 1.1)^15 - 1 = 0.99^15 - 1 < 0
        chunk = [-0.10, 0.10] * 15
        # 6개월 연속 같은 패턴 → 6 consecutive loss months → RETIRED
        returns = chunk * 6  # 180일
        _inject_daily_returns(pod, returns)

        retirement = RetirementCriteria(consecutive_loss_months=6)
        mgr = _make_manager(retirement=retirement)
        mgr.evaluate(pod)

        # 기하 복리로 계산하면 모든 월이 손실 → 6개월 연속 손실 → RETIRED
        assert pod.state == LifecycleState.RETIRED


# ── TestDistributionDrift ───────────────────────────────────────


class TestDistributionDrift:
    def test_set_and_get_distribution_result(self) -> None:
        """Distribution detector 설정 + evaluate 후 결과 조회."""
        pod = _make_pod()
        pod.state = LifecycleState.PRODUCTION
        _inject_daily_returns(pod, [0.005] * 50)

        mgr = _make_manager()
        mgr.set_distribution_reference(pod.pod_id, [0.005] * 200, window_size=40)

        # Evaluate triggers _check_degradation → dist_detector.update
        mgr.evaluate(pod)

        result = mgr.get_distribution_result(pod.pod_id)
        assert result is not None
        # Same distribution → should not drift
        from src.monitoring.anomaly.distribution import DriftSeverity

        assert result.severity == DriftSeverity.NORMAL

    def test_dist_result_none_without_detector(self) -> None:
        """Detector 미설정 → None."""
        mgr = _make_manager()
        assert mgr.get_distribution_result("nonexistent") is None

    def test_dist_serialization(self) -> None:
        """to_dict/restore_from_dict에 dist_detector 포함."""
        pod = _make_pod()
        pod.state = LifecycleState.PRODUCTION
        _inject_daily_returns(pod, [0.005] * 5)

        mgr = _make_manager()
        mgr.set_distribution_reference(pod.pod_id, [0.005] * 100)
        mgr.evaluate(pod)

        state = mgr.to_dict()
        assert "dist_detector" in state[pod.pod_id]


# ── TestConformalRANSAC ─────────────────────────────────────────


class TestConformalRANSAC:
    def test_set_and_get_ransac_result(self) -> None:
        """RANSAC detector 설정 + evaluate 후 결과 조회."""
        pod = _make_pod()
        pod.state = LifecycleState.PRODUCTION
        _inject_daily_returns(pod, [0.005] * 50)

        mgr = _make_manager()
        mgr.set_ransac_params(pod.pod_id, window_size=100, alpha=0.05)

        mgr.evaluate(pod)

        result = mgr.get_ransac_result(pod.pod_id)
        assert result is not None
        from src.monitoring.anomaly.conformal_ransac import DecaySeverity

        # min_samples=60 > 50 returns → NORMAL (min samples 미달)
        # But actually the latest return is fed, so only 1 sample
        assert result.severity == DecaySeverity.NORMAL

    def test_ransac_result_none_without_detector(self) -> None:
        """Detector 미설정 → None."""
        mgr = _make_manager()
        assert mgr.get_ransac_result("nonexistent") is None

    def test_ransac_serialization(self) -> None:
        """to_dict/restore_from_dict에 ransac_detector 포함."""
        pod = _make_pod()
        pod.state = LifecycleState.PRODUCTION
        _inject_daily_returns(pod, [0.005] * 5)

        mgr = _make_manager()
        mgr.set_ransac_params(pod.pod_id)
        mgr.evaluate(pod)

        state = mgr.to_dict()
        assert "ransac_detector" in state[pod.pod_id]


# ── TestAutoInitDetectors ─────────────────────────────────────


class TestAutoInitDetectors:
    """Step 6: Degradation 검출기 자동 초기화 테스트."""

    def test_auto_init_detectors_sets_all_three(self) -> None:
        """3개 검출기 모두 non-None."""
        import numpy as np

        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 100).tolist()

        mgr = _make_manager()
        mgr.auto_init_detectors("pod-a", returns)

        pls = mgr._pod_states["pod-a"]
        assert pls.gbm_monitor is not None
        assert pls.dist_detector is not None
        assert pls.ransac_detector is not None

    def test_auto_init_detectors_correct_params(self) -> None:
        """mu/sigma 값 검증."""
        import numpy as np

        returns = [0.01, 0.02, -0.01, 0.005, 0.015]
        expected_mu = float(np.mean(returns))
        expected_sigma = float(np.std(returns, ddof=1))

        mgr = _make_manager()
        mgr.auto_init_detectors("pod-a", returns)

        pls = mgr._pod_states["pod-a"]
        assert pls.gbm_monitor is not None
        assert pls.gbm_monitor._mu == pytest.approx(expected_mu)
        assert pls.gbm_monitor._sigma == pytest.approx(expected_sigma)

    def test_auto_init_detectors_insufficient_data(self) -> None:
        """returns < 2 → skip."""
        mgr = _make_manager()
        mgr.auto_init_detectors("pod-a", [0.01])

        # Pod state should not have detectors
        pls = mgr._pod_states.get("pod-a")
        if pls is not None:
            assert pls.gbm_monitor is None

    def test_auto_init_detectors_zero_volatility(self) -> None:
        """동일 값 → skip (sigma ≈ 0)."""
        mgr = _make_manager()
        mgr.auto_init_detectors("pod-a", [0.01, 0.01, 0.01, 0.01])

        pls = mgr._pod_states.get("pod-a")
        if pls is not None:
            assert pls.gbm_monitor is None

    def test_auto_init_detectors_idempotent(self) -> None:
        """2회 호출 → 에러 없음."""
        import numpy as np

        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 50).tolist()

        mgr = _make_manager()
        mgr.auto_init_detectors("pod-a", returns)
        mgr.auto_init_detectors("pod-a", returns)

        pls = mgr._pod_states["pod-a"]
        assert pls.gbm_monitor is not None
