"""Tests for OrchestratorMetrics — Prometheus Gauge/Enum 검증."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, PropertyMock

import pandas as pd
import pytest
from prometheus_client import REGISTRY

from src.orchestrator.metrics import OrchestratorMetrics
from src.orchestrator.models import LifecycleState, PodPerformance


def _sample(name: str, labels: dict[str, str] | None = None) -> float | None:
    """REGISTRY에서 특정 metric 값을 가져옴."""
    return REGISTRY.get_sample_value(name, labels or {})


def _make_mock_pod(
    pod_id: str,
    *,
    equity: float = 10000.0,
    fraction: float = 0.5,
    sharpe: float = 1.5,
    drawdown: float = 0.05,
    state: LifecycleState = LifecycleState.PRODUCTION,
    is_active: bool = True,
    daily_returns: list[float] | None = None,
) -> MagicMock:
    """Mock StrategyPod 생성."""
    pod = MagicMock()
    pod.pod_id = pod_id
    pod.capital_fraction = fraction
    pod.state = state
    pod.is_active = is_active
    pod.performance = PodPerformance(
        pod_id=pod_id,
        current_equity=equity,
        sharpe_ratio=sharpe,
        current_drawdown=drawdown,
    )
    returns = daily_returns if daily_returns is not None else [0.01, -0.005, 0.008]
    pod.daily_returns_series = pd.Series(returns, dtype=float)
    return pod


def _make_mock_orchestrator(
    pods: list[MagicMock] | None = None,
    initial_capital: float = 25000.0,
) -> MagicMock:
    """Mock StrategyOrchestrator 생성."""
    orch = MagicMock()
    if pods is None:
        pods = [
            _make_mock_pod("pod-a", equity=1.0, fraction=0.6, sharpe=1.8, drawdown=0.03),
            _make_mock_pod("pod-b", equity=1.0, fraction=0.4, sharpe=1.2, drawdown=0.07),
        ]
    type(orch).pods = PropertyMock(return_value=pods)
    type(orch).active_pod_count = PropertyMock(return_value=sum(1 for p in pods if p.is_active))
    type(orch).initial_capital = PropertyMock(return_value=initial_capital)
    return orch


class TestPodEquityGauge:
    def test_pod_equity_gauge_updated(self) -> None:
        """Pod equity gauge가 올바르게 업데이트되는지 확인."""
        orch = _make_mock_orchestrator()
        metrics = OrchestratorMetrics(orch)
        metrics.update()

        # initial_capital(25000) * fraction * current_equity(1.0)
        assert _sample("mcbot_pod_equity_usdt", {"pod_id": "pod-a"}) == pytest.approx(15000.0)
        assert _sample("mcbot_pod_equity_usdt", {"pod_id": "pod-b"}) == pytest.approx(10000.0)


class TestPodAllocationGauge:
    def test_pod_allocation_gauge_updated(self) -> None:
        """Pod allocation fraction gauge가 올바르게 업데이트되는지 확인."""
        orch = _make_mock_orchestrator()
        metrics = OrchestratorMetrics(orch)
        metrics.update()

        assert _sample("mcbot_pod_allocation_fraction", {"pod_id": "pod-a"}) == pytest.approx(0.6)
        assert _sample("mcbot_pod_allocation_fraction", {"pod_id": "pod-b"}) == pytest.approx(0.4)


class TestPodLifecycleState:
    def test_pod_lifecycle_state_updated(self) -> None:
        """Pod lifecycle enum이 올바른 상태로 설정되는지 확인."""
        pods = [
            _make_mock_pod("pod-x", state=LifecycleState.PRODUCTION),
            _make_mock_pod("pod-y", state=LifecycleState.WARNING),
        ]
        orch = _make_mock_orchestrator(pods)
        metrics = OrchestratorMetrics(orch)
        metrics.update()

        # Enum metric: mcbot_pod_lifecycle_state{pod_id="pod-x", mcbot_pod_lifecycle_state="production"} = 1.0
        assert (
            _sample(
                "mcbot_pod_lifecycle_state",
                {"pod_id": "pod-x", "mcbot_pod_lifecycle_state": "production"},
            )
            == 1.0
        )
        assert (
            _sample(
                "mcbot_pod_lifecycle_state",
                {"pod_id": "pod-y", "mcbot_pod_lifecycle_state": "warning"},
            )
            == 1.0
        )


class TestPodSharpeGauge:
    def test_pod_sharpe_gauge_updated(self) -> None:
        """Pod sharpe gauge가 올바르게 업데이트되는지 확인."""
        orch = _make_mock_orchestrator()
        metrics = OrchestratorMetrics(orch)
        metrics.update()

        assert _sample("mcbot_pod_rolling_sharpe", {"pod_id": "pod-a"}) == pytest.approx(1.8)
        assert _sample("mcbot_pod_rolling_sharpe", {"pod_id": "pod-b"}) == pytest.approx(1.2)


class TestPodDrawdownGauge:
    def test_pod_drawdown_gauge_updated(self) -> None:
        """Pod drawdown gauge가 올바르게 업데이트되는지 확인."""
        orch = _make_mock_orchestrator()
        metrics = OrchestratorMetrics(orch)
        metrics.update()

        assert _sample("mcbot_pod_drawdown_pct", {"pod_id": "pod-a"}) == pytest.approx(0.03)
        assert _sample("mcbot_pod_drawdown_pct", {"pod_id": "pod-b"}) == pytest.approx(0.07)


class TestPodRiskContribution:
    def test_pod_risk_contribution_updated(self) -> None:
        """Pod PRC gauge가 업데이트되는지 확인."""
        orch = _make_mock_orchestrator()
        metrics = OrchestratorMetrics(orch)
        metrics.update()

        prc_a = _sample("mcbot_pod_risk_contribution", {"pod_id": "pod-a"})
        prc_b = _sample("mcbot_pod_risk_contribution", {"pod_id": "pod-b"})
        assert prc_a is not None
        assert prc_b is not None
        # PRC 합 ≈ 1.0
        assert prc_a + prc_b == pytest.approx(1.0, abs=0.05)


class TestPortfolioEffectiveN:
    def test_portfolio_effective_n_gauge(self) -> None:
        """Portfolio effective N gauge가 업데이트되는지 확인."""
        orch = _make_mock_orchestrator()
        metrics = OrchestratorMetrics(orch)
        metrics.update()

        eff_n = _sample("mcbot_portfolio_effective_n")
        assert eff_n is not None
        assert eff_n > 0.0


class TestPortfolioAvgCorrelation:
    def test_portfolio_avg_correlation_gauge(self) -> None:
        """Portfolio avg correlation gauge가 업데이트되는지 확인."""
        orch = _make_mock_orchestrator()
        metrics = OrchestratorMetrics(orch)
        metrics.update()

        avg_corr = _sample("mcbot_portfolio_avg_correlation")
        assert avg_corr is not None


class TestActivePods:
    def test_active_pods_gauge(self) -> None:
        """Active pods gauge가 올바르게 업데이트되는지 확인."""
        orch = _make_mock_orchestrator()
        metrics = OrchestratorMetrics(orch)
        metrics.update()

        assert _sample("mcbot_active_pods") == 2.0


class TestSinglePodEdgeCase:
    def test_single_pod_skips_portfolio_metrics(self) -> None:
        """Pod < 2일 때 portfolio 메트릭이 안전하게 처리되는지 확인."""
        pods = [_make_mock_pod("pod-solo", fraction=1.0)]
        orch = _make_mock_orchestrator(pods)
        metrics = OrchestratorMetrics(orch)
        metrics.update()

        # single pod → effective_n = 1, avg_corr = 0
        assert _sample("mcbot_portfolio_effective_n") == pytest.approx(1.0)
        assert _sample("mcbot_portfolio_avg_correlation") == pytest.approx(0.0)
        # PRC = 1.0 (균등 배분)
        assert _sample("mcbot_pod_risk_contribution", {"pod_id": "pod-solo"}) == pytest.approx(1.0)


class TestNettingGauges:
    def test_netting_gauges_updated(self) -> None:
        """Netting 관련 Prometheus Gauge 업데이트 검증."""
        orch = _make_mock_orchestrator()
        # Set up last_pod_targets with opposing positions
        type(orch).last_pod_targets = PropertyMock(
            return_value={
                "pod-a": {"BTC/USDT": 0.3, "ETH/USDT": 0.2},
                "pod-b": {"BTC/USDT": -0.1, "ETH/USDT": 0.1},
            }
        )
        metrics = OrchestratorMetrics(orch)
        metrics.update()

        gross = _sample("mcbot_netting_gross_exposure")
        net = _sample("mcbot_netting_net_exposure")
        offset = _sample("mcbot_netting_offset_ratio")

        assert gross is not None
        assert net is not None
        assert offset is not None
        # gross = 0.3 + 0.2 + 0.1 + 0.1 = 0.7
        assert gross == pytest.approx(0.7)
        # net: BTC = 0.2, ETH = 0.3 → |0.2| + |0.3| = 0.5
        assert net == pytest.approx(0.5)
        # offset = 1 - 0.5/0.7
        assert offset == pytest.approx(1.0 - 0.5 / 0.7)

    def test_netting_gauges_empty_targets(self) -> None:
        """last_pod_targets가 비어 있으면 0으로 설정."""
        orch = _make_mock_orchestrator()
        type(orch).last_pod_targets = PropertyMock(return_value={})
        metrics = OrchestratorMetrics(orch)
        metrics.update()

        assert _sample("mcbot_netting_gross_exposure") == pytest.approx(0.0)
        assert _sample("mcbot_netting_net_exposure") == pytest.approx(0.0)
        assert _sample("mcbot_netting_offset_ratio") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Anomaly Metrics
# ---------------------------------------------------------------------------


@dataclass
class _FakeDriftResult:
    """DriftCheckResult 대체."""

    ks_statistic: float
    p_value: float


@dataclass
class _FakeDecayResult:
    """DecayCheckResult 대체."""

    ransac_slope: float
    conformal_lower_bound: float
    slope_positive: bool
    level_breach: bool
    current_cumulative: float = 0.0


@dataclass
class _FakeDrawdownResult:
    """DrawdownCheckResult 대체."""

    current_depth: float = 0.0
    current_duration_days: int = 0
    severity: object = None  # DrawdownSeverity stub

    def __post_init__(self) -> None:
        if self.severity is None:
            self.severity = _FakeSeverity("NORMAL")


@dataclass
class _FakeSeverity:
    """Enum-like severity stub."""

    value: str


class TestAnomalyMetrics:
    def test_anomaly_gauges_updated(self) -> None:
        """Anomaly detection 결과가 Prometheus gauge에 반영되는지 확인."""
        pods = [
            _make_mock_pod("pod-a", is_active=True),
            _make_mock_pod("pod-b", is_active=True),
        ]
        orch = _make_mock_orchestrator(pods)

        # Mock lifecycle
        lifecycle = MagicMock()
        lifecycle.get_distribution_result.side_effect = lambda pid: {
            "pod-a": _FakeDriftResult(ks_statistic=0.15, p_value=0.03),
            "pod-b": _FakeDriftResult(ks_statistic=0.08, p_value=0.45),
        }.get(pid)
        lifecycle.get_ransac_result.side_effect = lambda pid: {
            "pod-a": _FakeDecayResult(
                ransac_slope=0.002,
                conformal_lower_bound=-0.01,
                slope_positive=True,
                level_breach=False,
                current_cumulative=0.05,
            ),
            "pod-b": _FakeDecayResult(
                ransac_slope=-0.001,
                conformal_lower_bound=0.005,
                slope_positive=False,
                level_breach=True,
                current_cumulative=-0.02,
            ),
        }.get(pid)
        lifecycle.get_gbm_result.side_effect = lambda pid: {
            "pod-a": _FakeDrawdownResult(
                current_depth=0.05,
                current_duration_days=3,
                severity=_FakeSeverity("NORMAL"),
            ),
            "pod-b": _FakeDrawdownResult(
                current_depth=0.15,
                current_duration_days=10,
                severity=_FakeSeverity("WARNING"),
            ),
        }.get(pid)
        type(orch).lifecycle = PropertyMock(return_value=lifecycle)

        metrics = OrchestratorMetrics(orch)
        metrics.update()

        # Distribution gauges
        assert _sample("mcbot_distribution_ks_statistic", {"strategy": "pod-a"}) == pytest.approx(
            0.15
        )
        assert _sample("mcbot_distribution_p_value", {"strategy": "pod-a"}) == pytest.approx(0.03)
        assert _sample("mcbot_distribution_ks_statistic", {"strategy": "pod-b"}) == pytest.approx(
            0.08
        )
        assert _sample("mcbot_distribution_p_value", {"strategy": "pod-b"}) == pytest.approx(0.45)

        # RANSAC gauges
        assert _sample("mcbot_ransac_slope", {"strategy": "pod-a"}) == pytest.approx(0.002)
        assert _sample("mcbot_ransac_conformal_lower", {"strategy": "pod-a"}) == pytest.approx(
            -0.01
        )
        # pod-a: slope_positive=True, level_breach=False → decay=0.0
        assert _sample("mcbot_ransac_decay_detected", {"strategy": "pod-a"}) == pytest.approx(0.0)

        # pod-b: slope_positive=False → decay=1.0
        assert _sample("mcbot_ransac_slope", {"strategy": "pod-b"}) == pytest.approx(-0.001)
        assert _sample("mcbot_ransac_decay_detected", {"strategy": "pod-b"}) == pytest.approx(1.0)

        # RANSAC cumulative
        assert _sample("mcbot_ransac_current_cumulative", {"strategy": "pod-a"}) == pytest.approx(
            0.05
        )
        assert _sample("mcbot_ransac_current_cumulative", {"strategy": "pod-b"}) == pytest.approx(
            -0.02
        )

        # GBM drawdown gauges
        assert _sample("mcbot_gbm_drawdown_depth", {"strategy": "pod-a"}) == pytest.approx(0.05)
        assert _sample("mcbot_gbm_drawdown_duration_days", {"strategy": "pod-a"}) == pytest.approx(
            3.0
        )
        assert _sample("mcbot_gbm_severity", {"strategy": "pod-a"}) == pytest.approx(0.0)

        assert _sample("mcbot_gbm_drawdown_depth", {"strategy": "pod-b"}) == pytest.approx(0.15)
        assert _sample("mcbot_gbm_drawdown_duration_days", {"strategy": "pod-b"}) == pytest.approx(
            10.0
        )
        assert _sample("mcbot_gbm_severity", {"strategy": "pod-b"}) == pytest.approx(1.0)

    def test_anomaly_skips_inactive_pods(self) -> None:
        """비활성 Pod는 anomaly gauge 갱신 스킵."""
        pods = [_make_mock_pod("pod-inactive", is_active=False)]
        orch = _make_mock_orchestrator(pods)

        lifecycle = MagicMock()
        lifecycle.get_distribution_result.return_value = _FakeDriftResult(
            ks_statistic=0.99, p_value=0.001
        )
        type(orch).lifecycle = PropertyMock(return_value=lifecycle)

        metrics = OrchestratorMetrics(orch)
        metrics.update()

        # 비활성이므로 lifecycle 메서드 호출 안 됨
        lifecycle.get_distribution_result.assert_not_called()

    def test_anomaly_none_lifecycle_safe(self) -> None:
        """lifecycle이 None이면 안전하게 스킵."""
        orch = _make_mock_orchestrator()
        type(orch).lifecycle = PropertyMock(return_value=None)

        metrics = OrchestratorMetrics(orch)
        # 에러 없이 완료
        metrics.update()

    def test_anomaly_none_results_safe(self) -> None:
        """detector 결과가 None이면 gauge 갱신 스킵."""
        pods = [_make_mock_pod("pod-new", is_active=True)]
        orch = _make_mock_orchestrator(pods)

        lifecycle = MagicMock()
        lifecycle.get_distribution_result.return_value = None
        lifecycle.get_ransac_result.return_value = None
        lifecycle.get_gbm_result.return_value = None
        type(orch).lifecycle = PropertyMock(return_value=lifecycle)

        metrics = OrchestratorMetrics(orch)
        # 에러 없이 완료
        metrics.update()
