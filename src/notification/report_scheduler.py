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

from src.notification.formatters import (
    format_bar_close_report_embed,
    format_daily_report_embed,
    format_enhanced_daily_report_embed,
    format_spot_daily_report_embed,
    format_spot_monthly_report_embed,
    format_spot_weekly_report_embed,
    format_weekly_report_embed,
)
from src.notification.models import ChannelRoute, NotificationItem, Severity

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.eda.analytics import AnalyticsEngine
    from src.eda.portfolio_manager import EDAPortfolioManager
    from src.monitoring.chart_generator import ChartGenerator
    from src.notification.health_collector import HealthDataCollector
    from src.notification.queue import NotificationQueue

# 월요일 = 0 (Python datetime.weekday())
_MONDAY = 0
# Bar close 후 데이터 안정화 대기 (초)
_BAR_CLOSE_DELAY_SECONDS = 120


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
    ) -> None:
        self._queue = queue
        self._analytics = analytics
        self._chart_gen = chart_gen
        self._pm = pm
        self._health_collector = health_collector
        self._daily_task: asyncio.Task[None] | None = None
        self._weekly_task: asyncio.Task[None] | None = None
        self._monthly_task: asyncio.Task[None] | None = None
        self._bar_close_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """스케줄 task 시작."""
        self._daily_task = asyncio.create_task(self._daily_loop())
        self._weekly_task = asyncio.create_task(self._weekly_loop())
        self._monthly_task = asyncio.create_task(self._monthly_loop())
        self._bar_close_task = asyncio.create_task(self._bar_close_loop())
        logger.info("ReportScheduler started (daily + weekly + monthly + bar_close)")

    async def stop(self) -> None:
        """Task 취소."""
        for task in (self._daily_task, self._weekly_task, self._monthly_task, self._bar_close_task):
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
        """매일 00:00 UTC 일일 리포트 (상위 리포트 있는 날은 생략)."""
        while True:
            await _sleep_until_next(hour=0, minute=0)
            # 우선순위: Monthly > Weekly > Daily
            now = datetime.now(UTC)
            if now.day == 1:
                logger.info("Daily report skipped (Monthly report day)")
                continue
            if now.weekday() == _MONDAY:
                logger.info("Daily report skipped (Weekly report day)")
                continue
            try:
                await self._send_daily_report()
            except Exception:
                logger.exception("Failed to send daily report")

    async def _bar_close_loop(self) -> None:
        """매일 12:00 UTC (21:00 KST) bar close 리포트."""
        while True:
            await _sleep_until_next(hour=12, minute=0)
            # Bar close 후 데이터 안정화 대기
            await asyncio.sleep(_BAR_CLOSE_DELAY_SECONDS)
            try:
                await self._send_bar_close_report()
            except Exception:
                logger.exception("Failed to send bar close report")

    async def _send_bar_close_report(self) -> None:
        """12H bar close 리포트 생성 + enqueue."""
        if self._health_collector is None:
            logger.warning("Bar close report skipped: no health_collector")
            return
        if getattr(self._health_collector, "_spot_client", None) is None:
            logger.warning("Bar close report skipped: no spot_client")
            return

        report_data = await self._health_collector.collect_bar_close_report_data("12:00")
        embed = format_bar_close_report_embed(report_data)

        severity = Severity.WARNING if report_data.is_circuit_breaker_active else Severity.INFO
        item = NotificationItem(
            severity=severity,
            channel=ChannelRoute.DAILY_REPORT,
            embed=embed,
        )
        await self._queue.enqueue(item)
        logger.info("Bar close report enqueued")

    async def _weekly_loop(self) -> None:
        """매주 월요일 00:00 UTC 주간 리포트 (매월 1일은 생략)."""
        while True:
            await _sleep_until_next(hour=0, minute=0, weekday=_MONDAY)
            now = datetime.now(UTC)
            if now.day == 1:
                logger.info("Weekly report skipped (Monthly report day)")
                continue
            try:
                await self._send_weekly_report()
            except Exception:
                logger.exception("Failed to send weekly report")

    async def _monthly_loop(self) -> None:
        """매월 1일 00:00 UTC 월간 리포트."""
        while True:
            await _sleep_until_next_monthly()
            try:
                await self._send_monthly_report()
            except Exception:
                logger.exception("Failed to send monthly report")

    async def _send_monthly_report(self) -> None:
        """월간 리포트 생성 + enqueue."""
        equity_series = self._analytics.get_equity_series()
        trades = self._analytics.closed_trades
        metrics = self._analytics.compute_metrics()

        now = datetime.now(UTC)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        trades_month = [t for t in trades if t.exit_time and t.exit_time >= month_start]

        embed: dict[str, object]
        severity = Severity.INFO
        _has_spot = (
            self._health_collector is not None
            and getattr(self._health_collector, "_spot_client", None) is not None
        )
        if _has_spot:
            assert self._health_collector is not None
            try:
                report_data = await self._health_collector.collect_monthly_report_data()
                embed = format_spot_monthly_report_embed(report_data)
                if report_data.alpha_decay_detected:
                    severity = Severity.WARNING
            except Exception:
                logger.exception("Spot monthly report failed, falling back to weekly")
                embed = format_weekly_report_embed(metrics=metrics, trades_week=trades_month)
        else:
            embed = format_weekly_report_embed(metrics=metrics, trades_week=trades_month)

        # 차트 생성
        loop = asyncio.get_running_loop()
        charts = await loop.run_in_executor(
            None, self._chart_gen.generate_weekly_report, equity_series, trades_month, metrics,
        )

        files = tuple(charts)
        item = NotificationItem(
            severity=severity,
            channel=ChannelRoute.DAILY_REPORT,
            embed=embed,
            files=files,
        )
        await self._queue.enqueue(item)
        logger.info("Monthly report enqueued ({} charts)", len(files))

    async def _send_daily_report(self) -> None:
        """일일 리포트 생성 + enqueue."""
        equity_series = self._analytics.get_equity_series()
        trades = self._analytics.closed_trades
        metrics = self._analytics.compute_metrics()

        today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        trades_today = [t for t in trades if t.exit_time and t.exit_time >= today_start]

        # Spot Daily Report (새 경로: spot_client 있을 때)
        embed: dict[str, object]
        severity = Severity.INFO
        _has_spot_report = (
            self._health_collector is not None
            and getattr(self._health_collector, "_spot_client", None) is not None
        )
        if _has_spot_report:
            assert self._health_collector is not None  # narrowing for pyright
            try:
                report_data = await self._health_collector.collect_daily_report_data()
                embed = format_spot_daily_report_embed(report_data)
                if report_data.alpha_decay_detected:
                    severity = Severity.WARNING
            except Exception:
                logger.exception("Spot daily report failed, falling back to legacy")
                embed, severity = self._build_legacy_embed(metrics, trades_today)
        else:
            embed, severity = self._build_legacy_embed(metrics, trades_today)

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
        logger.info("Daily report enqueued ({} charts)", len(files))

    def _build_legacy_embed(
        self,
        metrics: object,
        trades_today: Sequence[object],
    ) -> tuple[dict[str, object], Severity]:
        """Legacy daily report embed (fallback).

        Returns:
            (embed dict, severity) 튜플
        """
        system_health = None
        strategy_health = None
        if self._health_collector is not None:
            try:
                system_health = self._health_collector.collect_system_health()
                strategy_health = self._health_collector.collect_strategy_health()
            except Exception:
                logger.exception("Failed to collect health data for daily report")

        severity = Severity.INFO
        if strategy_health is not None and strategy_health.alpha_decay_detected:
            severity = Severity.WARNING

        has_health = system_health is not None or strategy_health is not None
        if has_health:
            embed = format_enhanced_daily_report_embed(
                metrics=metrics,  # type: ignore[arg-type]
                open_positions=self._pm.open_position_count,
                total_equity=self._pm.total_equity,
                trades_today=trades_today,  # type: ignore[arg-type]
                system_health=system_health,
                strategy_health=strategy_health,
            )
            return embed, severity
        return (
            format_daily_report_embed(
                metrics=metrics,  # type: ignore[arg-type]
                open_positions=self._pm.open_position_count,
                total_equity=self._pm.total_equity,
                trades_today=trades_today,  # type: ignore[arg-type]
            ),
            severity,
        )

    async def _send_weekly_report(self) -> None:
        """주간 리포트 생성 + enqueue."""
        equity_series = self._analytics.get_equity_series()
        trades = self._analytics.closed_trades
        metrics = self._analytics.compute_metrics()

        now = datetime.now(UTC)
        week_start = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        trades_week = [t for t in trades if t.exit_time and t.exit_time >= week_start]

        # Spot Weekly Report (spot_client 있을 때)
        embed: dict[str, object]
        severity = Severity.INFO
        _has_spot = (
            self._health_collector is not None
            and getattr(self._health_collector, "_spot_client", None) is not None
        )
        if _has_spot:
            assert self._health_collector is not None
            try:
                report_data = await self._health_collector.collect_weekly_report_data()
                embed = format_spot_weekly_report_embed(report_data)
                if report_data.alpha_decay_detected:
                    severity = Severity.WARNING
            except Exception:
                logger.exception("Spot weekly report failed, falling back to legacy")
                embed = format_weekly_report_embed(metrics=metrics, trades_week=trades_week)
        else:
            embed = format_weekly_report_embed(metrics=metrics, trades_week=trades_week)

        # 차트 생성
        loop = asyncio.get_running_loop()
        charts = await loop.run_in_executor(
            None, self._chart_gen.generate_weekly_report, equity_series, trades_week, metrics,
        )

        files = tuple(charts)
        item = NotificationItem(
            severity=severity,
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


async def _sleep_until_next_monthly() -> None:
    """다음 달 1일 00:00 UTC까지 sleep."""
    now = datetime.now(UTC)
    if now.month == 12:  # noqa: PLR2004
        target = now.replace(year=now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        target = now.replace(month=now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)

    wait_seconds = (target - now).total_seconds()
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)
