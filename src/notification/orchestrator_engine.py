"""OrchestratorNotificationEngine — Orchestrator → Discord 알림.

Orchestrator 이벤트(생애주기 전이, 리밸런스, 리스크 경고)를
NotificationQueue에 enqueue하는 fire-and-forget 엔진입니다.

Daily report는 ReportScheduler로 통합되어 여기서는 생성하지 않습니다.

Rules Applied:
    - EDA 패턴: asyncio task lifecycle, fire-and-forget
    - #10 Python Standards: Async patterns, type hints
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from src.notification.models import ChannelRoute, NotificationItem, Severity
from src.notification.orchestrator_formatters import (
    format_capital_rebalance_embed,
    format_lifecycle_transition_embed,
    format_portfolio_risk_alert_embed,
)

if TYPE_CHECKING:
    from src.notification.queue import NotificationQueue
    from src.orchestrator.models import RiskAlert
    from src.orchestrator.orchestrator import StrategyOrchestrator
    from src.orchestrator.surveillance import ScanResult

# ── Constants ────────────────────────────────────────────────────

_LIFECYCLE_SEVERITY: dict[str, Severity] = {
    "retired": Severity.CRITICAL,
    "probation": Severity.WARNING,
    "warning": Severity.WARNING,
    "production": Severity.INFO,
    "incubation": Severity.INFO,
}


class OrchestratorNotificationEngine:
    """Orchestrator 이벤트 → Discord 알림 엔진.

    Daily report는 ReportScheduler에서 통합 생성합니다.
    이 엔진은 lifecycle/rebalance/risk/surveillance 알림만 담당합니다.

    Args:
        queue: NotificationQueue 인스턴스
        orchestrator: StrategyOrchestrator 인스턴스
    """

    def __init__(
        self,
        queue: NotificationQueue,
        orchestrator: StrategyOrchestrator,
    ) -> None:
        self._queue = queue
        self._orchestrator = orchestrator

    async def start(self) -> None:
        """엔진 시작."""
        logger.info("OrchestratorNotificationEngine started")

    async def stop(self) -> None:
        """엔진 중지."""
        logger.info("OrchestratorNotificationEngine stopped")

    # ── Public notify methods ────────────────────────────────────

    async def notify_lifecycle_transition(
        self,
        pod_id: str,
        from_state: str,
        to_state: str,
        timestamp: str,
        performance_summary: dict[str, object] | None = None,
    ) -> None:
        """생애주기 상태 전이 알림 → ALERTS 채널."""
        embed = format_lifecycle_transition_embed(
            pod_id=pod_id,
            from_state=from_state,
            to_state=to_state,
            timestamp=timestamp,
            performance_summary=performance_summary,
        )
        severity = _LIFECYCLE_SEVERITY.get(to_state, Severity.INFO)
        item = NotificationItem(
            severity=severity,
            channel=ChannelRoute.ALERTS,
            embed=embed,
        )
        await self._queue.enqueue(item)

    async def notify_capital_rebalance(
        self,
        timestamp: str,
        allocations: dict[str, float],
        trigger_reason: str,
    ) -> None:
        """자본 리밸런스 알림 → ALERTS 채널."""
        embed = format_capital_rebalance_embed(
            timestamp=timestamp,
            allocations=allocations,
            trigger_reason=trigger_reason,
        )
        item = NotificationItem(
            severity=Severity.INFO,
            channel=ChannelRoute.ALERTS,
            embed=embed,
            spam_key="orchestrator_rebalance",
        )
        await self._queue.enqueue(item)

    async def notify_risk_alerts(self, alerts: list[RiskAlert]) -> None:
        """리스크 경고 알림 → ALERTS 채널."""
        if not alerts:
            return

        for alert in alerts:
            embed = format_portfolio_risk_alert_embed(
                alert_type=alert.alert_type,
                severity=alert.severity,
                message=alert.message,
                current_value=alert.current_value,
                threshold=alert.threshold,
                pod_id=alert.pod_id,
            )
            severity = Severity.CRITICAL if alert.severity == "critical" else Severity.WARNING
            item = NotificationItem(
                severity=severity,
                channel=ChannelRoute.ALERTS,
                embed=embed,
                spam_key=f"orch_risk:{alert.alert_type}",
            )
            await self._queue.enqueue(item)

    # ── Surveillance ──────────────────────────────────────────────

    async def on_surveillance_scan(
        self,
        scan_result: ScanResult,
        pod_additions: dict[str, list[str]],
    ) -> None:
        """Surveillance 스캔 결과 Discord 알림.

        변경 없는 스캔은 알림을 발행하지 않습니다.

        Args:
            scan_result: ScanResult 인스턴스
            pod_additions: Pod별 추가된 심볼
        """
        if not scan_result.added and not scan_result.dropped:
            return

        from src.notification.formatters import format_surveillance_scan_embed

        embed = format_surveillance_scan_embed(scan_result, pod_additions)
        item = NotificationItem(
            severity=Severity.INFO,
            channel=ChannelRoute.ALERTS,
            embed=embed,
            spam_key="surveillance_scan",
        )
        await self._queue.enqueue(item)
