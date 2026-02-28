"""ReportScheduler — 주기적 일일/주간 리포트 스케줄러.

asyncio 기반으로 Daily (00:00 UTC) 및 Weekly (월요일 00:00 UTC) 리포트를
생성하여 NotificationQueue에 enqueue합니다.

Rules Applied:
    - EDA 패턴: asyncio task lifecycle
    - #10 Python Standards: Async patterns, type hints
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.notification.formatters import (
    format_daily_report_embed,
    format_unified_daily_report_embed,
    format_weekly_report_embed,
)
from src.notification.models import ChannelRoute, NotificationItem, Severity

if TYPE_CHECKING:
    from src.eda.analytics import AnalyticsEngine
    from src.eda.portfolio_manager import EDAPortfolioManager
    from src.monitoring.chart_generator import ChartGenerator
    from src.notification.health_collector import HealthDataCollector
    from src.notification.queue import NotificationQueue
    from src.orchestrator.orchestrator import StrategyOrchestrator

# 월요일 = 0 (Python datetime.weekday())
_MONDAY = 0


class ReportScheduler:
    """주기적 리포트 스케줄러.

    Args:
        queue: NotificationQueue (enqueue 대상)
        analytics: AnalyticsEngine (메트릭 소스)
        chart_gen: ChartGenerator (차트 생성)
        pm: EDAPortfolioManager (포지션 정보)
    """

    def __init__(
        self,
        queue: NotificationQueue,
        analytics: AnalyticsEngine,
        chart_gen: ChartGenerator,
        pm: EDAPortfolioManager,
        health_collector: HealthDataCollector | None = None,
        orchestrator: StrategyOrchestrator | None = None,
    ) -> None:
        self._queue = queue
        self._analytics = analytics
        self._chart_gen = chart_gen
        self._pm = pm
        self._health_collector = health_collector
        self._orchestrator = orchestrator
        self._daily_task: asyncio.Task[None] | None = None
        self._weekly_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """스케줄 task 시작."""
        self._daily_task = asyncio.create_task(self._daily_loop())
        self._weekly_task = asyncio.create_task(self._weekly_loop())
        logger.info("ReportScheduler started (daily + weekly)")

    async def stop(self) -> None:
        """Task 취소."""
        for task in (self._daily_task, self._weekly_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        logger.info("ReportScheduler stopped")

    async def trigger_daily_report(self) -> None:
        """즉시 daily report 생성 + enqueue (Discord /report 명령용)."""
        try:
            await self._send_daily_report()
            logger.info("Daily report triggered manually")
        except Exception:
            logger.exception("Failed to trigger daily report")

    async def _daily_loop(self) -> None:
        """매일 00:00 UTC 일일 리포트."""
        while True:
            await _sleep_until_next(hour=0, minute=0)
            try:
                await self._send_daily_report()
            except Exception:
                logger.exception("Failed to send daily report")

    async def _weekly_loop(self) -> None:
        """매주 월요일 00:00 UTC 주간 리포트."""
        while True:
            await _sleep_until_next(hour=0, minute=0, weekday=_MONDAY)
            try:
                await self._send_weekly_report()
            except Exception:
                logger.exception("Failed to send weekly report")

    async def _send_daily_report(self) -> None:
        """일일 리포트 생성 + enqueue (health + orchestrator 데이터 통합)."""
        equity_series = self._analytics.get_equity_series()
        trades = self._analytics.closed_trades
        metrics = self._analytics.compute_metrics()

        # 오늘 거래 필터
        today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        trades_today = [t for t in trades if t.exit_time and t.exit_time >= today_start]

        # Health 데이터 수집 (collector 있을 때만)
        system_health = None
        strategy_health = None
        regime_report = None
        if self._health_collector is not None:
            try:
                system_health = self._health_collector.collect_system_health()
                strategy_health = self._health_collector.collect_strategy_health()
                regime_report = await self._health_collector.collect_regime_report()
            except Exception:
                logger.exception("Failed to collect health data for daily report")

        # Orchestrator 데이터 수집
        orch_kwargs = self._collect_orchestrator_data()

        # Embed 결정
        has_health = (
            system_health is not None or strategy_health is not None or regime_report is not None
        )
        if has_health or orch_kwargs:
            embed = format_unified_daily_report_embed(
                metrics=metrics,
                open_positions=self._pm.open_position_count,
                total_equity=self._pm.total_equity,
                trades_today=trades_today,
                system_health=system_health,
                strategy_health=strategy_health,
                regime_report=regime_report,
                **orch_kwargs,
            )
        else:
            embed = format_daily_report_embed(
                metrics=metrics,
                open_positions=self._pm.open_position_count,
                total_equity=self._pm.total_equity,
                trades_today=trades_today,
            )

        # Severity — alpha decay 시 WARNING
        severity = Severity.INFO
        if strategy_health is not None and strategy_health.alpha_decay_detected:
            severity = Severity.WARNING

        # 차트 생성 (blocking → executor)
        loop = asyncio.get_running_loop()
        charts = await loop.run_in_executor(
            None, self._chart_gen.generate_daily_report, equity_series, trades_today, metrics
        )

        files = tuple(charts)
        item = NotificationItem(
            severity=severity,
            channel=ChannelRoute.DAILY_REPORT,
            embed=embed,
            files=files,
        )
        await self._queue.enqueue(item)
        logger.info(
            "Daily report enqueued ({} charts, orchestrator={})",
            len(files),
            self._orchestrator is not None,
        )

    def _collect_orchestrator_data(self) -> dict[str, Any]:
        """Orchestrator 포트폴리오 메트릭 수집.

        Returns:
            format_unified_daily_report_embed()에 전달할 kwargs dict.
            orchestrator가 없으면 빈 dict 반환.
        """
        if self._orchestrator is None:
            return {}

        try:
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

            # Pod returns + weights
            _min_pods_for_portfolio = 2
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

            if len(active_pods) >= _min_pods_for_portfolio:
                max_len = max(len(v) for v in pod_returns_data.values())
                for pid, current in pod_returns_data.items():
                    if len(current) < max_len:
                        pod_returns_data[pid] = [0.0] * (max_len - len(current)) + current

                pod_returns = pd.DataFrame(pod_returns_data)
                prc = compute_risk_contributions(pod_returns, weights)
                effective_n = compute_effective_n(prc)
                _, avg_correlation = check_correlation_stress(pod_returns, 1.0)

            portfolio_dd = compute_portfolio_drawdown(performances, weights)  # type: ignore[arg-type]
            gross_leverage = compute_gross_leverage({})
        except Exception:
            logger.exception("Failed to collect orchestrator data for daily report")
            return {}

        return {
            "pod_summaries": pod_summaries,
            "effective_n": effective_n,
            "avg_correlation": avg_correlation,
            "portfolio_dd": portfolio_dd,
            "gross_leverage": gross_leverage,
        }

    async def _send_weekly_report(self) -> None:
        """주간 리포트 생성 + enqueue."""
        equity_series = self._analytics.get_equity_series()
        trades = self._analytics.closed_trades
        metrics = self._analytics.compute_metrics()

        # 이번 주 거래 필터
        now = datetime.now(UTC)
        week_start = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        trades_week = [t for t in trades if t.exit_time and t.exit_time >= week_start]

        # Embed
        embed = format_weekly_report_embed(
            metrics=metrics,
            trades_week=trades_week,
        )

        # 차트 생성
        loop = asyncio.get_running_loop()
        charts = await loop.run_in_executor(
            None, self._chart_gen.generate_weekly_report, equity_series, trades_week, metrics
        )

        files = tuple(charts)
        item = NotificationItem(
            severity=Severity.INFO,
            channel=ChannelRoute.DAILY_REPORT,
            embed=embed,
            files=files,
        )
        await self._queue.enqueue(item)
        logger.info("Weekly report enqueued ({} charts)", len(files))


async def _sleep_until_next(
    hour: int,
    minute: int,
    weekday: int | None = None,
) -> None:
    """다음 스케줄 시점까지 sleep.

    Args:
        hour: 목표 시 (UTC)
        minute: 목표 분 (UTC)
        weekday: 목표 요일 (0=Monday, None이면 매일)
    """
    now = datetime.now(UTC)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    if weekday is not None:
        # 다음 weekday 계산
        days_ahead = weekday - now.weekday()
        if days_ahead < 0 or (days_ahead == 0 and now >= target):
            days_ahead += 7
        target = target + timedelta(days=days_ahead)
    elif now >= target:
        target = target + timedelta(days=1)

    wait_seconds = (target - now).total_seconds()
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)
