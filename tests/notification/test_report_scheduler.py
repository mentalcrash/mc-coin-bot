"""ReportScheduler 테스트."""

from __future__ import annotations

import asyncio
import contextlib
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

    # Mock HealthDataCollector (legacy 경로 — spot_client 없음)
    collector = MagicMock()
    collector._spot_client = None  # legacy fallback 유도
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

    scheduler = ReportScheduler(
        queue=queue,
        analytics=analytics,
        chart_gen=chart_gen,
        pm=pm,
        health_collector=collector,
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
        # enhanced는 fields가 6 + system status 등 추가
        assert len(item.embed["fields"]) > 6

    async def test_daily_report_enhanced_without_collector(self) -> None:
        """collector=None → 기존 embed fallback."""
        scheduler, queue = _make_scheduler()
        await scheduler._send_daily_report()

        item = queue.enqueue.call_args[0][0]
        assert item.embed["title"] == "Daily Report"
        assert len(item.embed["fields"]) == 6  # 기존 6개 필드

    async def test_daily_report_alpha_decay_severity(self) -> None:
        """alpha_decay → Severity.WARNING."""
        from src.notification.models import Severity

        scheduler, queue, collector = _make_scheduler_with_collector()
        strategy_health = collector.collect_strategy_health.return_value
        strategy_health.alpha_decay_detected = True

        await scheduler._send_daily_report()

        item = queue.enqueue.call_args[0][0]
        assert item.severity == Severity.WARNING


# ─── Bar Close Report 테스트 ────────────────────────────


def _make_scheduler_with_spot_collector() -> tuple[ReportScheduler, AsyncMock, MagicMock]:
    """spot_client 포함 ReportScheduler 생성 (Bar Close Report 용)."""
    from src.notification.health_models import (
        AssetDashboardItem,
        BarCloseReportData,
        SignalChangeItem,
    )

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
    chart_gen.generate_daily_report.return_value = [("equity.png", b"\x89PNG")]

    pm = MagicMock()
    pm.open_position_count = 2
    pm.total_equity = 10100.0

    collector = MagicMock()
    collector._spot_client = MagicMock()  # spot_client 존재 → bar close 활성화
    collector.collect_bar_close_report_data = AsyncMock(
        return_value=BarCloseReportData(
            bar_time_utc="12:00",
            signal_changes=(
                SignalChangeItem(
                    symbol="BTC/USDT",
                    prev_signal="NEUTRAL",
                    new_signal="LONG",
                    realized_pnl=None,
                ),
            ),
            assets=(
                AssetDashboardItem(
                    symbol="BTC/USDT",
                    signal="LONG",
                    current_price=42000.0,
                    change_24h_pct=2.5,
                    position_value=4200.0,
                    day_pnl=100.0,
                    stop_price=40000.0,
                    stop_distance_pct=4.8,
                ),
            ),
            total_equity=10100.0,
            today_pnl=100.0,
            invested_count=2,
            total_asset_count=6,
            uptime_seconds=86400.0,
            is_circuit_breaker_active=False,
            ws_ok_count=6,
            ws_total_count=6,
        )
    )

    scheduler = ReportScheduler(
        queue=queue,
        analytics=analytics,
        chart_gen=chart_gen,
        pm=pm,
        health_collector=collector,
    )
    return scheduler, queue, collector


class TestBarCloseReport:
    async def test_enqueues_bar_close_report(self) -> None:
        """Bar close report가 정상 enqueue."""
        scheduler, queue, _collector = _make_scheduler_with_spot_collector()
        await scheduler._send_bar_close_report()

        queue.enqueue.assert_called_once()
        item = queue.enqueue.call_args[0][0]
        assert "12H Bar Close" in item.embed["title"]
        assert item.files == ()

    async def test_bar_close_has_signal_changes(self) -> None:
        """Signal change 섹션 포함."""
        scheduler, queue, _collector = _make_scheduler_with_spot_collector()
        await scheduler._send_bar_close_report()

        item = queue.enqueue.call_args[0][0]
        signal_field = item.embed["fields"][0]
        assert signal_field["name"] == "Signal Changes"
        assert "BTC/USDT" in signal_field["value"]

    async def test_bar_close_skipped_without_spot_client(self) -> None:
        """spot_client 없으면 skip."""
        scheduler, queue, collector = _make_scheduler_with_spot_collector()
        collector._spot_client = None

        await scheduler._send_bar_close_report()
        queue.enqueue.assert_not_called()

    async def test_bar_close_skipped_without_collector(self) -> None:
        """health_collector 없으면 skip."""
        scheduler, queue = _make_scheduler()
        await scheduler._send_bar_close_report()
        queue.enqueue.assert_not_called()


# ─── Spot Weekly Report 테스트 ──────────────────────────


def _make_scheduler_with_spot_weekly() -> tuple[ReportScheduler, AsyncMock, MagicMock]:
    """spot_client 포함 ReportScheduler (Weekly Report 용)."""
    from src.notification.health_models import (
        AssetWeeklyPerformance,
        StrategyIndicatorItem,
        WeeklyReportData,
    )

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
    chart_gen.generate_weekly_report.return_value = [
        ("equity.png", b"\x89PNG"),
        ("drawdown.png", b"\x89PNG2"),
    ]

    pm = MagicMock()
    pm.open_position_count = 2
    pm.total_equity = 10100.0

    collector = MagicMock()
    collector._spot_client = MagicMock()
    collector.collect_weekly_report_data = AsyncMock(
        return_value=WeeklyReportData(
            strategy_name="SuperTrend",
            strategy_params={"ATR": "7", "mult": "2.5"},
            trailing_stop_config="3.0x ATR",
            timeframe="12h",
            total_equity=10100.0,
            available_cash=5000.0,
            cash_pct=49.5,
            week_pnl=150.0,
            week_trades=3,
            invested_count=2,
            total_asset_count=6,
            cumulative_return_pct=1.0,
            max_drawdown_pct=2.5,
            assets=(
                AssetWeeklyPerformance(
                    symbol="BTC/USDT",
                    signal="LONG",
                    current_price=42000.0,
                    week_change_pct=5.2,
                    week_pnl=120.0,
                    week_trades=2,
                ),
            ),
            best_trade_symbol="BTC/USDT",
            best_trade_pnl=80.0,
            worst_trade_symbol="ETH/USDT",
            worst_trade_pnl=-10.0,
            week_win_rate=0.67,
            week_profit_factor=8.0,
            indicators=(
                StrategyIndicatorItem(
                    symbol="BTC/USDT",
                    supertrend_line=40000.0,
                    adx_value=30.0,
                    outlook="상승추세",
                ),
            ),
            uptime_seconds=604800.0,
            is_circuit_breaker_active=False,
            ws_ok_count=6,
            ws_total_count=6,
            rolling_sharpe_30d=1.2,
            win_rate=0.6,
            profit_factor=1.8,
            alpha_decay_detected=False,
        )
    )

    scheduler = ReportScheduler(
        queue=queue,
        analytics=analytics,
        chart_gen=chart_gen,
        pm=pm,
        health_collector=collector,
    )
    return scheduler, queue, collector


class TestSpotWeeklyReport:
    async def test_enqueues_spot_weekly_report(self) -> None:
        """Spot weekly report 정상 enqueue."""
        scheduler, queue, _collector = _make_scheduler_with_spot_weekly()
        await scheduler._send_weekly_report()

        queue.enqueue.assert_called_once()
        item = queue.enqueue.call_args[0][0]
        assert item.embed["title"] == "Spot Weekly Report"
        assert len(item.files) == 2

    async def test_weekly_has_asset_performance(self) -> None:
        """에셋별 주간 성과 포함."""
        scheduler, queue, _collector = _make_scheduler_with_spot_weekly()
        await scheduler._send_weekly_report()

        item = queue.enqueue.call_args[0][0]
        field_names = [f["name"] for f in item.embed["fields"]]
        assert any("Asset Weekly" in n for n in field_names)

    async def test_weekly_has_trade_summary(self) -> None:
        """Best/Worst trade 포함."""
        scheduler, queue, _collector = _make_scheduler_with_spot_weekly()
        await scheduler._send_weekly_report()

        item = queue.enqueue.call_args[0][0]
        field_names = [f["name"] for f in item.embed["fields"]]
        assert "Best Trade" in field_names
        assert "Worst Trade" in field_names

    async def test_weekly_fallback_without_spot_client(self) -> None:
        """spot_client 없으면 legacy weekly fallback."""
        scheduler, queue, collector = _make_scheduler_with_spot_weekly()
        collector._spot_client = None

        await scheduler._send_weekly_report()

        item = queue.enqueue.call_args[0][0]
        assert item.embed["title"] == "Weekly Report"


class TestReportPriority:
    async def test_daily_skipped_on_monday(self) -> None:
        """월요일에는 daily loop가 skip."""
        scheduler, queue = _make_scheduler()

        monday = datetime(2026, 3, 9, 0, 0, 1, tzinfo=UTC)  # 월요일

        call_count = 0

        async def fake_sleep_until_next(**_kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise asyncio.CancelledError

        with (
            patch(
                "src.notification.report_scheduler._sleep_until_next",
                side_effect=fake_sleep_until_next,
            ),
            patch(
                "src.notification.report_scheduler.datetime",
            ) as mock_dt,
        ):
            mock_dt.now.return_value = monday
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            with contextlib.suppress(asyncio.CancelledError):
                await scheduler._daily_loop()

        queue.enqueue.assert_not_called()

    async def test_daily_skipped_on_first_of_month(self) -> None:
        """매월 1일에는 daily loop가 skip."""
        scheduler, queue = _make_scheduler()

        first_day = datetime(2026, 4, 1, 0, 0, 1, tzinfo=UTC)  # 4/1 수요일

        call_count = 0

        async def fake_sleep_until_next(**_kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise asyncio.CancelledError

        with (
            patch(
                "src.notification.report_scheduler._sleep_until_next",
                side_effect=fake_sleep_until_next,
            ),
            patch(
                "src.notification.report_scheduler.datetime",
            ) as mock_dt,
        ):
            mock_dt.now.return_value = first_day
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            with contextlib.suppress(asyncio.CancelledError):
                await scheduler._daily_loop()

        queue.enqueue.assert_not_called()

    async def test_weekly_skipped_on_first_of_month(self) -> None:
        """매월 1일이 월요일이면 weekly loop가 skip."""
        scheduler, queue, _collector = _make_scheduler_with_spot_weekly()

        # 2026-06-01 is Monday
        first_monday = datetime(2026, 6, 1, 0, 0, 1, tzinfo=UTC)

        call_count = 0

        async def fake_sleep_until_next(**_kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise asyncio.CancelledError

        with (
            patch(
                "src.notification.report_scheduler._sleep_until_next",
                side_effect=fake_sleep_until_next,
            ),
            patch(
                "src.notification.report_scheduler.datetime",
            ) as mock_dt,
        ):
            mock_dt.now.return_value = first_monday
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            with contextlib.suppress(asyncio.CancelledError):
                await scheduler._weekly_loop()

        queue.enqueue.assert_not_called()


# ─── Spot Monthly Report 테스트 ─────────────────────────


def _make_scheduler_with_spot_monthly() -> tuple[ReportScheduler, AsyncMock, MagicMock]:
    """spot_client 포함 ReportScheduler (Monthly Report 용)."""
    from src.notification.health_models import (
        AssetMonthlyPerformance,
        MonthlyPerformanceTrend,
        MonthlyReportData,
        StrategyIndicatorItem,
    )

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
    chart_gen.generate_weekly_report.return_value = [
        ("equity.png", b"\x89PNG"),
        ("drawdown.png", b"\x89PNG2"),
    ]

    pm = MagicMock()
    pm.open_position_count = 3
    pm.total_equity = 10500.0

    collector = MagicMock()
    collector._spot_client = MagicMock()
    collector.collect_monthly_report_data = AsyncMock(
        return_value=MonthlyReportData(
            strategy_name="SuperTrend",
            strategy_params={"ATR": "7", "mult": "2.5"},
            trailing_stop_config="3.0x ATR",
            timeframe="12h",
            total_equity=10500.0,
            available_cash=4000.0,
            cash_pct=38.1,
            month_pnl=500.0,
            month_trades=15,
            month_return_pct=5.0,
            invested_count=3,
            total_asset_count=6,
            cumulative_return_pct=5.0,
            max_drawdown_pct=3.0,
            assets=(
                AssetMonthlyPerformance(
                    symbol="BTC/USDT",
                    signal="LONG",
                    current_price=42000.0,
                    month_change_pct=12.3,
                    month_pnl=350.0,
                    month_trades=8,
                ),
            ),
            best_trade_symbol="BTC/USDT",
            best_trade_pnl=120.0,
            worst_trade_symbol="ETH/USDT",
            worst_trade_pnl=-30.0,
            month_win_rate=0.73,
            month_profit_factor=3.5,
            avg_trade_pnl=33.3,
            total_fees=15.0,
            performance_trend=(
                MonthlyPerformanceTrend(
                    year_month="2026-03",
                    pnl=500.0,
                    return_pct=5.0,
                    trades=15,
                    sharpe=1.1,
                ),
                MonthlyPerformanceTrend(
                    year_month="2026-02",
                    pnl=310.0,
                    return_pct=3.2,
                    trades=10,
                    sharpe=0.95,
                ),
            ),
            indicators=(
                StrategyIndicatorItem(
                    symbol="BTC/USDT",
                    supertrend_line=40000.0,
                    adx_value=30.0,
                    outlook="상승추세",
                ),
            ),
            uptime_seconds=2592000.0,
            is_circuit_breaker_active=False,
            ws_ok_count=6,
            ws_total_count=6,
            rolling_sharpe_30d=1.1,
            win_rate=0.65,
            profit_factor=2.0,
            alpha_decay_detected=False,
            month_max_drawdown_pct=1.5,
            longest_losing_streak=3,
        )
    )

    scheduler = ReportScheduler(
        queue=queue,
        analytics=analytics,
        chart_gen=chart_gen,
        pm=pm,
        health_collector=collector,
    )
    return scheduler, queue, collector


class TestSpotMonthlyReport:
    async def test_enqueues_spot_monthly_report(self) -> None:
        """Spot monthly report 정상 enqueue."""
        scheduler, queue, _collector = _make_scheduler_with_spot_monthly()
        await scheduler._send_monthly_report()

        queue.enqueue.assert_called_once()
        item = queue.enqueue.call_args[0][0]
        assert item.embed["title"] == "Spot Monthly Report"
        assert len(item.files) == 2

    async def test_monthly_has_performance_trend(self) -> None:
        """Performance Trend 섹션 포함."""
        scheduler, queue, _collector = _make_scheduler_with_spot_monthly()
        await scheduler._send_monthly_report()

        item = queue.enqueue.call_args[0][0]
        field_names = [f["name"] for f in item.embed["fields"]]
        assert "Performance Trend" in field_names

    async def test_monthly_has_risk_summary(self) -> None:
        """Risk Summary 섹션 포함."""
        scheduler, queue, _collector = _make_scheduler_with_spot_monthly()
        await scheduler._send_monthly_report()

        item = queue.enqueue.call_args[0][0]
        field_names = [f["name"] for f in item.embed["fields"]]
        assert "Month Max DD" in field_names
        assert "Longest Losing Streak" in field_names

    async def test_monthly_has_fees_and_avg_pnl(self) -> None:
        """Avg Trade PnL, Total Fees 포함."""
        scheduler, queue, _collector = _make_scheduler_with_spot_monthly()
        await scheduler._send_monthly_report()

        item = queue.enqueue.call_args[0][0]
        field_names = [f["name"] for f in item.embed["fields"]]
        assert "Avg Trade PnL" in field_names
        assert "Total Fees" in field_names

    async def test_monthly_fallback_without_spot_client(self) -> None:
        """spot_client 없으면 legacy fallback."""
        scheduler, queue, collector = _make_scheduler_with_spot_monthly()
        collector._spot_client = None

        await scheduler._send_monthly_report()

        item = queue.enqueue.call_args[0][0]
        assert item.embed["title"] == "Weekly Report"
