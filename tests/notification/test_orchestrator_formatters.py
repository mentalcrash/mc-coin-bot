"""Tests for Orchestrator Discord Embed formatters."""

from __future__ import annotations

from src.notification.orchestrator_formatters import (
    _COLOR_BLUE,
    _COLOR_GREEN,
    _COLOR_RED,
    _COLOR_YELLOW,
    format_capital_rebalance_embed,
    format_daily_orchestrator_report_embed,
    format_lifecycle_transition_embed,
    format_portfolio_risk_alert_embed,
)


class TestLifecycleTransitionEmbed:
    def test_graduation_green_color(self) -> None:
        """PRODUCTION 전이 시 GREEN 색상."""
        embed = format_lifecycle_transition_embed(
            pod_id="pod-a",
            from_state="incubation",
            to_state="production",
            timestamp="2026-02-14T00:00:00Z",
        )
        assert embed["color"] == _COLOR_GREEN
        assert "PRODUCTION" in embed["title"]

    def test_retirement_red_color(self) -> None:
        """RETIRED 전이 시 RED 색상."""
        embed = format_lifecycle_transition_embed(
            pod_id="pod-a",
            from_state="probation",
            to_state="retired",
            timestamp="2026-02-14T00:00:00Z",
        )
        assert embed["color"] == _COLOR_RED
        assert "RETIRED" in embed["title"]

    def test_warning_yellow_color(self) -> None:
        """WARNING 전이 시 YELLOW 색상."""
        embed = format_lifecycle_transition_embed(
            pod_id="pod-a",
            from_state="production",
            to_state="warning",
            timestamp="2026-02-14T00:00:00Z",
        )
        assert embed["color"] == _COLOR_YELLOW
        assert "WARNING" in embed["title"]

    def test_with_performance_summary(self) -> None:
        """Performance summary가 포함된 embed."""
        embed = format_lifecycle_transition_embed(
            pod_id="pod-a",
            from_state="incubation",
            to_state="production",
            timestamp="2026-02-14T00:00:00Z",
            performance_summary={"sharpe": 1.5, "mdd": 0.08},
        )
        # Performance 필드가 존재
        field_names = [f["name"] for f in embed["fields"]]
        assert "Performance" in field_names


class TestCapitalRebalanceEmbed:
    def test_capital_rebalance_embed_fields(self) -> None:
        """리밸런스 embed에 trigger, pods 필드 포함."""
        embed = format_capital_rebalance_embed(
            timestamp="2026-02-14T00:00:00Z",
            allocations={"pod-a": 0.6, "pod-b": 0.4},
            trigger_reason="calendar",
        )
        assert embed["color"] == _COLOR_BLUE
        assert "Rebalance" in embed["title"]
        field_names = [f["name"] for f in embed["fields"]]
        assert "Trigger" in field_names
        assert "Pods" in field_names
        # Description에 pod 이름 포함
        assert "pod-a" in embed["description"]
        assert "pod-b" in embed["description"]


class TestPortfolioRiskAlertEmbed:
    def test_portfolio_risk_alert_embed(self) -> None:
        """리스크 경고 embed 구조 검증."""
        embed = format_portfolio_risk_alert_embed(
            alert_type="gross_leverage",
            severity="critical",
            message="Gross leverage 3.2x vs limit 3.0x",
            current_value=3.2,
            threshold=3.0,
            pod_id=None,
        )
        assert embed["color"] == _COLOR_RED
        assert "gross_leverage" in embed["title"]
        field_names = [f["name"] for f in embed["fields"]]
        assert "Type" in field_names
        assert "Severity" in field_names
        # pod_id=None이면 Pod 필드 없음
        assert "Pod" not in field_names

    def test_warning_severity_yellow(self) -> None:
        """Warning severity 시 YELLOW 색상."""
        embed = format_portfolio_risk_alert_embed(
            alert_type="correlation_stress",
            severity="warning",
            message="Approaching threshold",
            current_value=0.55,
            threshold=0.70,
        )
        assert embed["color"] == _COLOR_YELLOW

    def test_with_pod_id(self) -> None:
        """pod_id 지정 시 Pod 필드 포함."""
        embed = format_portfolio_risk_alert_embed(
            alert_type="single_pod_risk",
            severity="critical",
            message="Pod pod-a PRC 50%",
            current_value=0.50,
            threshold=0.40,
            pod_id="pod-a",
        )
        field_names = [f["name"] for f in embed["fields"]]
        assert "Pod" in field_names

    def test_has_timestamp(self) -> None:
        """timestamp가 비어있지 않은 ISO 문자열."""
        embed = format_portfolio_risk_alert_embed(
            alert_type="test",
            severity="warning",
            message="test",
            current_value=0.5,
            threshold=1.0,
        )
        assert "timestamp" in embed
        assert len(embed["timestamp"]) > 0


class TestDailyOrchestratorReportEmbed:
    def test_daily_orchestrator_report_embed_table(self) -> None:
        """일일 리포트 embed 구조 검증."""
        summaries = [
            {"pod_id": "pod-a", "state": "production", "capital_fraction": 0.6, "live_days": 30},
            {"pod_id": "pod-b", "state": "incubation", "capital_fraction": 0.4, "live_days": 5},
        ]
        embed = format_daily_orchestrator_report_embed(
            pod_summaries=summaries,
            total_equity=100000.0,
            effective_n=1.8,
            avg_correlation=0.35,
            portfolio_dd=0.03,
            gross_leverage=1.2,
        )
        assert embed["color"] == _COLOR_GREEN  # dd < 5%
        assert "pod-a" in embed["description"]
        assert "pod-b" in embed["description"]
        field_names = [f["name"] for f in embed["fields"]]
        assert "Total Equity" in field_names
        assert "Effective N" in field_names
        assert "Avg Correlation" in field_names

    def test_has_timestamp(self) -> None:
        """timestamp가 비어있지 않은 ISO 문자열."""
        embed = format_daily_orchestrator_report_embed(
            pod_summaries=[],
            total_equity=100000.0,
            effective_n=1.0,
            avg_correlation=0.0,
            portfolio_dd=0.01,
            gross_leverage=1.0,
        )
        assert "timestamp" in embed
        assert len(embed["timestamp"]) > 0

    def test_high_drawdown_red_color(self) -> None:
        """DD > 10%이면 RED."""
        embed = format_daily_orchestrator_report_embed(
            pod_summaries=[],
            total_equity=80000.0,
            effective_n=0.0,
            avg_correlation=0.0,
            portfolio_dd=0.12,
            gross_leverage=1.0,
        )
        assert embed["color"] == _COLOR_RED
