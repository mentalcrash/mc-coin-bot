"""Tests for OrchestratorMetrics — Prometheus Gauge/Enum 검증."""

from __future__ import annotations

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
) -> MagicMock:
    """Mock StrategyOrchestrator 생성."""
    orch = MagicMock()
    if pods is None:
        pods = [
            _make_mock_pod("pod-a", equity=15000.0, fraction=0.6, sharpe=1.8, drawdown=0.03),
            _make_mock_pod("pod-b", equity=10000.0, fraction=0.4, sharpe=1.2, drawdown=0.07),
        ]
    type(orch).pods = PropertyMock(return_value=pods)
    type(orch).active_pod_count = PropertyMock(return_value=sum(1 for p in pods if p.is_active))
    return orch


class TestPodEquityGauge:
    def test_pod_equity_gauge_updated(self) -> None:
        """Pod equity gauge가 올바르게 업데이트되는지 확인."""
        orch = _make_mock_orchestrator()
        metrics = OrchestratorMetrics(orch)
        metrics.update()

        assert _sample("mcbot_pod_equity_usdt", {"pod_id": "pod-a"}) == 15000.0
        assert _sample("mcbot_pod_equity_usdt", {"pod_id": "pod-b"}) == 10000.0


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
