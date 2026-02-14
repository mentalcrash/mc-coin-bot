"""Tests for orchestrator domain models."""

from __future__ import annotations

from datetime import UTC, datetime

from src.orchestrator.models import (
    AllocationMethod,
    LifecycleState,
    PodPerformance,
    PodPosition,
    RebalanceTrigger,
    RiskAlert,
)

# ── LifecycleState ──────────────────────────────────────────────


class TestLifecycleState:
    def test_all_states_defined(self) -> None:
        assert len(LifecycleState) == 5

    def test_state_values(self) -> None:
        assert LifecycleState.INCUBATION == "incubation"
        assert LifecycleState.PRODUCTION == "production"
        assert LifecycleState.WARNING == "warning"
        assert LifecycleState.PROBATION == "probation"
        assert LifecycleState.RETIRED == "retired"


# ── AllocationMethod ────────────────────────────────────────────


class TestAllocationMethod:
    def test_all_methods_defined(self) -> None:
        assert len(AllocationMethod) == 4

    def test_method_values(self) -> None:
        assert AllocationMethod.EQUAL_WEIGHT == "equal_weight"
        assert AllocationMethod.RISK_PARITY == "risk_parity"
        assert AllocationMethod.ADAPTIVE_KELLY == "adaptive_kelly"
        assert AllocationMethod.INVERSE_VOLATILITY == "inverse_volatility"


# ── RebalanceTrigger ────────────────────────────────────────────


class TestRebalanceTrigger:
    def test_all_triggers_defined(self) -> None:
        assert len(RebalanceTrigger) == 3

    def test_trigger_values(self) -> None:
        assert RebalanceTrigger.CALENDAR == "calendar"
        assert RebalanceTrigger.THRESHOLD == "threshold"
        assert RebalanceTrigger.HYBRID == "hybrid"


# ── PodPerformance ──────────────────────────────────────────────


class TestPodPerformance:
    def test_default_values(self) -> None:
        perf = PodPerformance(pod_id="pod-1")
        assert perf.pod_id == "pod-1"
        assert perf.total_return == 0.0
        assert perf.sharpe_ratio == 0.0
        assert perf.max_drawdown == 0.0
        assert perf.trade_count == 0
        assert perf.live_days == 0
        assert isinstance(perf.last_updated, datetime)
        assert perf.last_updated.tzinfo == UTC

    def test_mutable_update(self) -> None:
        perf = PodPerformance(pod_id="pod-1")
        perf.total_return = 0.15
        perf.trade_count = 42
        assert perf.total_return == 0.15
        assert perf.trade_count == 42

    def test_is_profitable_true(self) -> None:
        perf = PodPerformance(pod_id="pod-1", total_return=0.05)
        assert perf.is_profitable is True

    def test_is_profitable_false(self) -> None:
        perf = PodPerformance(pod_id="pod-1", total_return=-0.03)
        assert perf.is_profitable is False

    def test_equity_ratio(self) -> None:
        perf = PodPerformance(pod_id="pod-1", peak_equity=10000.0, current_equity=8500.0)
        assert perf.equity_ratio == 0.85

    def test_equity_ratio_zero_peak(self) -> None:
        perf = PodPerformance(pod_id="pod-1", peak_equity=0.0, current_equity=0.0)
        assert perf.equity_ratio == 0.0


# ── PodPosition ─────────────────────────────────────────────────


class TestPodPosition:
    def test_default_values(self) -> None:
        pos = PodPosition(pod_id="pod-1", symbol="BTC/USDT")
        assert pos.pod_id == "pod-1"
        assert pos.symbol == "BTC/USDT"
        assert pos.target_weight == 0.0
        assert pos.global_weight == 0.0
        assert pos.notional_usd == 0.0
        assert pos.unrealized_pnl == 0.0
        assert pos.realized_pnl == 0.0

    def test_total_pnl(self) -> None:
        pos = PodPosition(
            pod_id="pod-1",
            symbol="BTC/USDT",
            unrealized_pnl=100.0,
            realized_pnl=50.0,
        )
        assert pos.total_pnl == 150.0

    def test_is_open(self) -> None:
        pos = PodPosition(pod_id="pod-1", symbol="BTC/USDT", notional_usd=5000.0)
        assert pos.is_open is True

    def test_is_not_open(self) -> None:
        pos = PodPosition(pod_id="pod-1", symbol="BTC/USDT", notional_usd=0.0)
        assert pos.is_open is False


# ── RiskAlert ──────────────────────────────────────────────────


class TestRiskAlert:
    def test_frozen(self) -> None:
        alert = RiskAlert(
            alert_type="gross_leverage",
            severity="critical",
            message="Leverage exceeded",
            current_value=3.5,
            threshold=3.0,
        )
        assert alert.alert_type == "gross_leverage"
        assert alert.severity == "critical"
        assert alert.pod_id is None

    def test_pod_id_set(self) -> None:
        alert = RiskAlert(
            alert_type="single_pod_risk",
            severity="warning",
            message="Pod risk high",
            current_value=0.35,
            threshold=0.40,
            pod_id="pod-a",
        )
        assert alert.pod_id == "pod-a"

    def test_has_critical_true(self) -> None:
        alerts = [
            RiskAlert("daily_loss", "warning", "Approaching limit", 0.02, 0.03),
            RiskAlert("gross_leverage", "critical", "Exceeded", 3.5, 3.0),
        ]
        assert RiskAlert.has_critical(alerts) is True

    def test_has_critical_false(self) -> None:
        alerts = [
            RiskAlert("daily_loss", "warning", "Approaching limit", 0.02, 0.03),
        ]
        assert RiskAlert.has_critical(alerts) is False

    def test_has_critical_empty(self) -> None:
        assert RiskAlert.has_critical([]) is False
