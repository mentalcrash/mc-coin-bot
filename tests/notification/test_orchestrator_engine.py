"""Tests for OrchestratorNotificationEngine — AsyncMock queue 패턴."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.notification.models import ChannelRoute, Severity
from src.notification.orchestrator_engine import OrchestratorNotificationEngine
from src.orchestrator.models import RiskAlert


def _make_engine() -> tuple[OrchestratorNotificationEngine, AsyncMock]:
    """AsyncMock queue로 engine 생성."""
    mock_queue = AsyncMock()
    mock_orch = MagicMock()
    engine = OrchestratorNotificationEngine(mock_queue, mock_orch)
    return engine, mock_queue


class TestLifecycleTransition:
    async def test_lifecycle_transition_enqueued_to_alerts(self) -> None:
        """생애주기 전이 알림이 ALERTS 채널에 enqueue됨."""
        engine, mock_queue = _make_engine()

        await engine.notify_lifecycle_transition(
            pod_id="pod-a",
            from_state="incubation",
            to_state="production",
            timestamp="2026-02-14T00:00:00Z",
        )

        mock_queue.enqueue.assert_called_once()
        item = mock_queue.enqueue.call_args[0][0]
        assert item.channel == ChannelRoute.ALERTS
        assert "PRODUCTION" in item.embed["title"]

    @pytest.mark.parametrize(
        ("to_state", "expected_severity"),
        [
            ("retired", Severity.CRITICAL),
            ("warning", Severity.WARNING),
            ("probation", Severity.WARNING),
            ("production", Severity.INFO),
            ("incubation", Severity.INFO),
        ],
    )
    async def test_lifecycle_severity_mapping(
        self, to_state: str, expected_severity: Severity
    ) -> None:
        """생애주기 상태별 severity 매핑."""
        engine, mock_queue = _make_engine()

        await engine.notify_lifecycle_transition(
            pod_id="pod-a",
            from_state="production",
            to_state=to_state,
            timestamp="2026-02-14T00:00:00Z",
        )

        item = mock_queue.enqueue.call_args[0][0]
        assert item.severity == expected_severity


class TestCapitalRebalance:
    async def test_capital_rebalance_enqueued_to_trade_log(self) -> None:
        """리밸런스 알림이 TRADE_LOG 채널에 enqueue됨."""
        engine, mock_queue = _make_engine()

        await engine.notify_capital_rebalance(
            timestamp="2026-02-14T00:00:00Z",
            allocations={"pod-a": 0.6, "pod-b": 0.4},
            trigger_reason="calendar",
        )

        mock_queue.enqueue.assert_called_once()
        item = mock_queue.enqueue.call_args[0][0]
        assert item.channel == ChannelRoute.TRADE_LOG
        assert item.spam_key == "orchestrator_rebalance"


class TestRiskAlerts:
    async def test_risk_alerts_enqueued_multiple(self) -> None:
        """여러 리스크 경고가 각각 enqueue됨."""
        engine, mock_queue = _make_engine()

        alerts = [
            RiskAlert(
                alert_type="gross_leverage",
                severity="critical",
                message="Leverage too high",
                current_value=3.5,
                threshold=3.0,
            ),
            RiskAlert(
                alert_type="correlation_stress",
                severity="warning",
                message="Correlation approaching",
                current_value=0.60,
                threshold=0.70,
            ),
        ]

        await engine.notify_risk_alerts(alerts)

        assert mock_queue.enqueue.call_count == 2
        # 첫 번째: critical
        item1 = mock_queue.enqueue.call_args_list[0][0][0]
        assert item1.severity == Severity.CRITICAL
        assert item1.channel == ChannelRoute.ALERTS
        # 두 번째: warning
        item2 = mock_queue.enqueue.call_args_list[1][0][0]
        assert item2.severity == Severity.WARNING

    async def test_risk_alerts_empty_no_enqueue(self) -> None:
        """빈 alerts 리스트이면 enqueue 호출 없음."""
        engine, mock_queue = _make_engine()

        await engine.notify_risk_alerts([])

        mock_queue.enqueue.assert_not_called()


def _patch_report_deps():
    """_send_daily_report 내부 의존 함수들을 patch하는 컨텍스트 매니저."""
    return (
        patch(
            "src.orchestrator.risk_aggregator.compute_portfolio_drawdown",
            return_value=0.05,
        ),
        patch(
            "src.orchestrator.netting.compute_gross_leverage",
            return_value=1.0,
        ),
    )


class TestDailyReportEquity:
    """Daily report Total Equity가 PM.total_equity를 사용하는지 검증."""

    async def test_daily_report_uses_pm_total_equity(self) -> None:
        """PM 주입 시 PM.total_equity 값이 embed에 반영됨."""
        mock_queue = AsyncMock()
        mock_orch = MagicMock()

        # Orchestrator 설정
        mock_pod = MagicMock()
        mock_pod.is_active = True
        mock_pod.pod_id = "pod-a"
        mock_pod.capital_fraction = 0.5
        mock_pod.daily_returns_series = [0.01, 0.02]
        mock_pod.performance.current_equity = 1.03
        mock_orch.pods = [mock_pod]
        mock_orch.initial_capital = 10_000.0
        mock_orch.get_pod_summary.return_value = [
            {
                "pod_id": "pod-a",
                "strategy": "test",
                "state": "production",
                "return_pct": 3.0,
                "sharpe": 1.5,
                "mdd_pct": 5.0,
            }
        ]

        # PM mock: 실시간 MTM equity
        mock_pm = MagicMock()
        mock_pm.total_equity = 12_345.67

        engine = OrchestratorNotificationEngine(mock_queue, mock_orch, pm=mock_pm)
        p1, p2 = _patch_report_deps()
        with p1, p2:
            await engine._send_daily_report()

        mock_queue.enqueue.assert_called_once()
        item = mock_queue.enqueue.call_args[0][0]
        # PM equity ($12,346)가 embed에 포함되어야 함
        assert "$12,346" in str(item.embed)

    async def test_daily_report_fallback_without_pm(self) -> None:
        """PM 미주입 시 Pod 기반 fallback 계산 사용."""
        mock_queue = AsyncMock()
        mock_orch = MagicMock()

        mock_pod = MagicMock()
        mock_pod.is_active = True
        mock_pod.pod_id = "pod-a"
        mock_pod.capital_fraction = 1.0
        mock_pod.daily_returns_series = [0.10]
        mock_pod.performance.current_equity = 1.10
        mock_orch.pods = [mock_pod]
        mock_orch.initial_capital = 10_000.0
        mock_orch.get_pod_summary.return_value = [
            {
                "pod_id": "pod-a",
                "strategy": "test",
                "state": "production",
                "return_pct": 10.0,
                "sharpe": 1.0,
                "mdd_pct": 3.0,
            }
        ]

        # PM 없이 생성
        engine = OrchestratorNotificationEngine(mock_queue, mock_orch)
        p1, p2 = _patch_report_deps()
        with p1, p2:
            await engine._send_daily_report()

        mock_queue.enqueue.assert_called_once()
        item = mock_queue.enqueue.call_args[0][0]
        # Fallback: 10000 * 1.0 * 1.10 = $11,000
        assert "$11,000" in str(item.embed)
