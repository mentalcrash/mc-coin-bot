"""ReportScheduler 테스트."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

from src.models.backtest import TradeRecord
from src.notification.report_scheduler import ReportScheduler, _sleep_until_next


def _make_trade(exit_time: datetime) -> TradeRecord:
    return TradeRecord(
        entry_time=exit_time - timedelta(hours=1),
        exit_time=exit_time,
        symbol="BTC/USDT",
        direction="LONG",
        entry_price=Decimal(40000),
        exit_price=Decimal(40100),
        size=Decimal("0.1"),
        pnl=Decimal("10.00"),
        pnl_pct=0.025,
        fees=Decimal("2.00"),
    )


def _make_scheduler() -> tuple[ReportScheduler, AsyncMock]:
    """테스트용 ReportScheduler + mock queue."""
    queue = AsyncMock()
    queue.enqueue = AsyncMock()

    analytics = MagicMock()
    analytics.get_equity_series.return_value = pd.Series(
        [10000.0, 10050.0, 10100.0],
        index=pd.date_range("2025-01-01", periods=3, freq="D", tz=UTC),
        dtype=float,
    )
    now = datetime.now(UTC)
    analytics.closed_trades = [_make_trade(now - timedelta(hours=1))]

    metrics_mock = MagicMock()
    metrics_mock.sharpe_ratio = 1.5
    metrics_mock.max_drawdown = 5.0
    metrics_mock.total_return = 10.0
    analytics.compute_metrics.return_value = metrics_mock

    chart_gen = MagicMock()
    chart_gen.generate_daily_report.return_value = [("equity.png", b"\x89PNG_fake")]
    chart_gen.generate_weekly_report.return_value = [
        ("equity.png", b"\x89PNG_fake"),
        ("drawdown.png", b"\x89PNG_fake2"),
    ]

    pm = MagicMock()
    pm.open_position_count = 2
    pm.total_equity = 10100.0

    scheduler = ReportScheduler(queue=queue, analytics=analytics, chart_gen=chart_gen, pm=pm)
    return scheduler, queue


class TestSleepUntilNext:
    async def test_daily_future_today(self) -> None:
        """오늘 아직 안 지난 시각이면 오늘 대기."""
        now = datetime.now(UTC)
        # 1분 뒤를 목표로 설정
        target_minute = (now.minute + 1) % 60
        target_hour = now.hour if target_minute > now.minute else (now.hour + 1) % 24

        with patch(
            "src.notification.report_scheduler.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            await _sleep_until_next(hour=target_hour, minute=target_minute)
            mock_sleep.assert_called_once()
            # 대기 시간이 0~3600초 사이
            wait = mock_sleep.call_args[0][0]
            assert 0 < wait <= 3600

    async def test_daily_past_today(self) -> None:
        """오늘 이미 지난 시각이면 내일 대기."""
        now = datetime.now(UTC)
        past_hour = (now.hour - 1) % 24

        with patch(
            "src.notification.report_scheduler.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            await _sleep_until_next(hour=past_hour, minute=0)
            mock_sleep.assert_called_once()
            wait = mock_sleep.call_args[0][0]
            # 약 23시간 이상 대기
            assert wait > 3600

    async def test_weekly_next_monday(self) -> None:
        """다음 월요일까지 대기."""
        with patch(
            "src.notification.report_scheduler.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            await _sleep_until_next(hour=0, minute=0, weekday=0)
            mock_sleep.assert_called_once()
            wait = mock_sleep.call_args[0][0]
            # 최대 7일 = 604800초
            assert 0 < wait <= 604800


class TestSendDailyReport:
    async def test_enqueues_daily_report(self) -> None:
        scheduler, queue = _make_scheduler()
        await scheduler._send_daily_report()

        queue.enqueue.assert_called_once()
        item = queue.enqueue.call_args[0][0]
        assert item.channel.value == "daily_report"
        assert item.embed["title"] == "Daily Report"
        assert len(item.files) == 1

    async def test_daily_report_has_charts(self) -> None:
        scheduler, queue = _make_scheduler()
        await scheduler._send_daily_report()

        item = queue.enqueue.call_args[0][0]
        filenames = [name for name, _ in item.files]
        assert "equity.png" in filenames


class TestSendWeeklyReport:
    async def test_enqueues_weekly_report(self) -> None:
        scheduler, queue = _make_scheduler()
        await scheduler._send_weekly_report()

        queue.enqueue.assert_called_once()
        item = queue.enqueue.call_args[0][0]
        assert item.embed["title"] == "Weekly Report"
        assert len(item.files) == 2

    async def test_weekly_report_has_multiple_charts(self) -> None:
        scheduler, queue = _make_scheduler()
        await scheduler._send_weekly_report()

        item = queue.enqueue.call_args[0][0]
        filenames = [name for name, _ in item.files]
        assert "equity.png" in filenames
        assert "drawdown.png" in filenames


class TestStartStop:
    async def test_start_creates_tasks(self) -> None:
        scheduler, _ = _make_scheduler()
        await scheduler.start()
        assert scheduler._daily_task is not None
        assert scheduler._weekly_task is not None
        await scheduler.stop()

    async def test_stop_cancels_tasks(self) -> None:
        scheduler, _ = _make_scheduler()
        await scheduler.start()
        await scheduler.stop()
        assert scheduler._daily_task is not None
        assert scheduler._daily_task.cancelled()


# ─── Enhanced Daily Report 테스트 ────────────────────────


def _make_scheduler_with_collector() -> tuple[ReportScheduler, AsyncMock, MagicMock]:
    """health_collector 포함 ReportScheduler 생성."""
    queue = AsyncMock()
    queue.enqueue = AsyncMock()

    analytics = MagicMock()
    analytics.get_equity_series.return_value = pd.Series(
        [10000.0, 10050.0, 10100.0],
        index=pd.date_range("2025-01-01", periods=3, freq="D", tz=UTC),
        dtype=float,
    )
    now = datetime.now(UTC)
    analytics.closed_trades = [_make_trade(now - timedelta(hours=1))]

    metrics_mock = MagicMock()
    metrics_mock.sharpe_ratio = 1.5
    metrics_mock.max_drawdown = 5.0
    metrics_mock.total_return = 10.0
    analytics.compute_metrics.return_value = metrics_mock

    chart_gen = MagicMock()
    chart_gen.generate_daily_report.return_value = [("equity.png", b"\x89PNG_fake")]

    pm = MagicMock()
    pm.open_position_count = 2
    pm.total_equity = 10100.0

    # Mock HealthDataCollector
    collector = MagicMock()
    system_health = MagicMock()
    system_health.uptime_seconds = 86400.0
    system_health.is_circuit_breaker_active = False
    system_health.total_symbols = 8
    system_health.stale_symbol_count = 0
    collector.collect_system_health.return_value = system_health

    strategy_health = MagicMock()
    strategy_health.alpha_decay_detected = False
    strategy_health.strategy_breakdown = ()
    strategy_health.open_positions = ()
    strategy_health.rolling_sharpe_30d = 1.2
    collector.collect_strategy_health.return_value = strategy_health

    regime_report = MagicMock()
    regime_report.regime_label = "Bullish"
    regime_report.regime_score = 0.35
    regime_report.symbols = ()
    collector.collect_regime_report = AsyncMock(return_value=regime_report)

    scheduler = ReportScheduler(
        queue=queue, analytics=analytics, chart_gen=chart_gen, pm=pm, health_collector=collector
    )
    return scheduler, queue, collector


class TestEnhancedDailyReport:
    async def test_daily_report_enhanced_with_health_data(self) -> None:
        """health_collector 있을 때 enhanced embed 사용."""
        scheduler, queue, _collector = _make_scheduler_with_collector()
        await scheduler._send_daily_report()

        queue.enqueue.assert_called_once()
        item = queue.enqueue.call_args[0][0]
        assert item.embed["title"] == "Daily Report"
        # enhanced는 fields가 6 + market regime 등 추가
        assert len(item.embed["fields"]) > 6

    async def test_daily_report_enhanced_without_collector(self) -> None:
        """collector=None → 기존 embed fallback."""
        scheduler, queue = _make_scheduler()
        await scheduler._send_daily_report()

        item = queue.enqueue.call_args[0][0]
        assert item.embed["title"] == "Daily Report"
        assert len(item.embed["fields"]) == 6  # 기존 6개 필드

    async def test_daily_report_enhanced_regime_none(self) -> None:
        """regime 실패 시 graceful."""
        scheduler, queue, collector = _make_scheduler_with_collector()
        collector.collect_regime_report = AsyncMock(return_value=None)

        await scheduler._send_daily_report()

        queue.enqueue.assert_called_once()
        item = queue.enqueue.call_args[0][0]
        assert item.embed["title"] == "Daily Report"
        # regime 없어도 enhanced embed (system_health가 있으므로)
        field_names = [f["name"] for f in item.embed["fields"]]
        assert "Market Regime" not in field_names

    async def test_daily_report_alpha_decay_severity(self) -> None:
        """alpha_decay → Severity.WARNING."""
        from src.notification.models import Severity

        scheduler, queue, collector = _make_scheduler_with_collector()
        strategy_health = collector.collect_strategy_health.return_value
        strategy_health.alpha_decay_detected = True

        await scheduler._send_daily_report()

        item = queue.enqueue.call_args[0][0]
        assert item.severity == Severity.WARNING
