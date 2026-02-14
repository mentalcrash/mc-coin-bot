"""Tests for OrchestratorNotificationEngine — AsyncMock queue 패턴."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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
