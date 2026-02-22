"""OrchestratorNotificationEngine — Orchestrator → Discord 알림.

Orchestrator 이벤트(생애주기 전이, 리밸런스, 리스크 경고)를
NotificationQueue에 enqueue하는 fire-and-forget 엔진입니다.

일일 Orchestrator 리포트도 스케줄링합니다 (00:05 UTC).

Rules Applied:
    - EDA 패턴: asyncio task lifecycle, fire-and-forget
    - #10 Python Standards: Async patterns, type hints
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from loguru import logger

from src.notification.models import ChannelRoute, NotificationItem, Severity
from src.notification.orchestrator_formatters import (
    format_capital_rebalance_embed,
    format_daily_orchestrator_report_embed,
    format_lifecycle_transition_embed,
    format_portfolio_risk_alert_embed,
)

if TYPE_CHECKING:
    from src.eda.portfolio_manager import EDAPortfolioManager
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

_REPORT_HOUR = 0
_REPORT_MINUTE = 5
_MIN_PODS_FOR_PORTFOLIO = 2


async def _sleep_until_target(hour: int, minute: int) -> None:
    """다음 스케줄 시점(매일 hour:minute UTC)까지 sleep."""
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now >= target:
        target = target + timedelta(days=1)
    wait_seconds = (target - now).total_seconds()
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)


class OrchestratorNotificationEngine:
    """Orchestrator 이벤트 → Discord 알림 엔진.

    Args:
        queue: NotificationQueue 인스턴스
        orchestrator: StrategyOrchestrator 인스턴스
    """

    def __init__(
        self,
        queue: NotificationQueue,
        orchestrator: StrategyOrchestrator,
        pm: EDAPortfolioManager | None = None,
    ) -> None:
        self._queue = queue
        self._orchestrator = orchestrator
        self._pm = pm
        self._report_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """일일 리포트 스케줄 task 시작."""
        self._report_task = asyncio.create_task(self._daily_report_loop())
        logger.info("OrchestratorNotificationEngine started")

    async def stop(self) -> None:
        """Task 취소."""
        if self._report_task is not None:
            self._report_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._report_task
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
        """자본 리밸런스 알림 → TRADE_LOG 채널."""
        embed = format_capital_rebalance_embed(
            timestamp=timestamp,
            allocations=allocations,
            trigger_reason=trigger_reason,
        )
        item = NotificationItem(
            severity=Severity.INFO,
            channel=ChannelRoute.TRADE_LOG,
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

    # ── Daily report ─────────────────────────────────────────────

    async def _daily_report_loop(self) -> None:
        """매일 00:05 UTC 일일 리포트."""
        while True:
            await _sleep_until_target(hour=_REPORT_HOUR, minute=_REPORT_MINUTE)
            try:
                await self._send_daily_report()
            except Exception:
                logger.exception("Failed to send orchestrator daily report")

    async def _send_daily_report(self) -> None:
        """Orchestrator 일일 리포트 생성 + enqueue."""
        import pandas as pd

        from src.orchestrator.netting import compute_gross_leverage
        from src.orchestrator.risk_aggregator import (
            check_correlation_stress,
            compute_effective_n,
            compute_portfolio_drawdown,
            compute_risk_contributions,
        )

        orch = self._orchestrator
        pod_summaries = orch.get_pod_summary()
        active_pods = [p for p in orch.pods if p.is_active]

        # Total equity: PM이 주입되면 실시간 MTM 사용 (Daily Report와 동일)
        if self._pm is not None:
            total_equity = self._pm.total_equity
        else:
            # Fallback: Pod daily-return 누적 곱으로 간접 환산
            cap = orch.initial_capital
            total_equity = sum(
                cap * p.capital_fraction * p.performance.current_equity for p in active_pods
            )

        # Pod returns + weights
        pod_returns_data: dict[str, list[float]] = {}
        weights: dict[str, float] = {}
        performances: dict[str, object] = {}
        for pod in active_pods:
            returns = list(pod.daily_returns_series)
            pod_returns_data[pod.pod_id] = returns if returns else [0.0]
            weights[pod.pod_id] = pod.capital_fraction
            performances[pod.pod_id] = pod.performance

        effective_n = 0.0
        avg_correlation = 0.0

        if len(active_pods) >= _MIN_PODS_FOR_PORTFOLIO:
            max_len = max(len(v) for v in pod_returns_data.values())
            for pid, current in pod_returns_data.items():
                if len(current) < max_len:
                    pod_returns_data[pid] = [0.0] * (max_len - len(current)) + current

            pod_returns = pd.DataFrame(pod_returns_data)
            prc = compute_risk_contributions(pod_returns, weights)
            effective_n = compute_effective_n(prc)
            _, avg_correlation = check_correlation_stress(pod_returns, 1.0)

        # Portfolio drawdown
        portfolio_dd = compute_portfolio_drawdown(performances, weights)  # type: ignore[arg-type]

        # Gross leverage (last net_weights unavailable → 0 fallback)
        gross_leverage = compute_gross_leverage({})

        embed = format_daily_orchestrator_report_embed(
            pod_summaries=pod_summaries,
            total_equity=total_equity,
            effective_n=effective_n,
            avg_correlation=avg_correlation,
            portfolio_dd=portfolio_dd,
            gross_leverage=gross_leverage,
        )

        item = NotificationItem(
            severity=Severity.INFO,
            channel=ChannelRoute.DAILY_REPORT,
            embed=embed,
        )
        await self._queue.enqueue(item)
        logger.info("Orchestrator daily report enqueued")
