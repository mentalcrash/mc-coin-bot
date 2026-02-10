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
from typing import TYPE_CHECKING

from loguru import logger

from src.notification.formatters import format_daily_report_embed, format_weekly_report_embed
from src.notification.models import ChannelRoute, NotificationItem, Severity

if TYPE_CHECKING:
    from src.eda.analytics import AnalyticsEngine
    from src.eda.portfolio_manager import EDAPortfolioManager
    from src.monitoring.chart_generator import ChartGenerator
    from src.notification.queue import NotificationQueue

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
    ) -> None:
        self._queue = queue
        self._analytics = analytics
        self._chart_gen = chart_gen
        self._pm = pm
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
        """일일 리포트 생성 + enqueue."""
        equity_series = self._analytics.get_equity_series()
        trades = self._analytics.closed_trades
        metrics = self._analytics.compute_metrics()

        # 오늘 거래 필터
        today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        trades_today = [t for t in trades if t.exit_time and t.exit_time >= today_start]

        # Embed
        embed = format_daily_report_embed(
            metrics=metrics,
            open_positions=self._pm.open_position_count,
            total_equity=self._pm.total_equity,
            trades_today=trades_today,
        )

        # 차트 생성 (blocking → executor)
        loop = asyncio.get_running_loop()
        charts = await loop.run_in_executor(
            None, self._chart_gen.generate_daily_report, equity_series, trades_today, metrics
        )

        files = tuple(charts)
        item = NotificationItem(
            severity=Severity.INFO,
            channel=ChannelRoute.DAILY_REPORT,
            embed=embed,
            files=files,
        )
        await self._queue.enqueue(item)
        logger.info("Daily report enqueued ({} charts)", len(files))

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
