"""HealthDataCollector 테스트."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.notification.health_collector import HealthDataCollector
from src.notification.health_models import (
    StrategyHealthSnapshot,
    SymbolDerivativesSnapshot,
    SystemHealthSnapshot,
)

# ─── Mock Factories ──────────────────────────────────────


def _make_mock_pm() -> MagicMock:
    pm = MagicMock()
    pm.total_equity = 50000.0
    pm.available_cash = 30000.0
    pm.aggregate_leverage = 0.4
    pm.open_position_count = 2
    pm.positions = {}
    return pm


def _make_mock_rm() -> MagicMock:
    rm = MagicMock()
    rm.current_drawdown = 0.03
    rm.peak_equity = 52000.0
    rm.is_circuit_breaker_active = False
    return rm


def _make_mock_analytics() -> MagicMock:
    analytics = MagicMock()
    analytics.closed_trades = []
    return analytics


def _make_mock_feed() -> MagicMock:
    feed = MagicMock()
    feed.stale_symbols = set()
    feed.bars_emitted = 500
    return feed


def _make_mock_bus() -> MagicMock:
    bus = MagicMock()
    bus.metrics = SimpleNamespace(events_dropped=0, max_queue_depth=10)
    return bus


def _make_mock_queue() -> MagicMock:
    queue = MagicMock()
    queue.is_degraded = False
    queue.enqueue = AsyncMock()
    return queue


def _make_collector(
    **overrides: object,
) -> HealthDataCollector:
    pm = overrides.get("pm") or _make_mock_pm()
    rm = overrides.get("rm") or _make_mock_rm()
    analytics = overrides.get("analytics") or _make_mock_analytics()
    feed = overrides.get("feed") or _make_mock_feed()
    bus = overrides.get("bus") or _make_mock_bus()
    queue = overrides.get("queue") or _make_mock_queue()
    symbols = overrides.get("symbols") or ["BTC/USDT", "ETH/USDT"]

    return HealthDataCollector(
        pm=pm,  # type: ignore[arg-type]
        rm=rm,  # type: ignore[arg-type]
        analytics=analytics,  # type: ignore[arg-type]
        feed=feed,  # type: ignore[arg-type]
        bus=bus,  # type: ignore[arg-type]
        queue=queue,  # type: ignore[arg-type]
        futures_client=None,
        symbols=symbols,  # type: ignore[arg-type]
    )


# ─── Collect System Health 테스트 ─────────────────────────


class TestCollectSystemHealth:
    def test_basic_snapshot(self) -> None:
        collector = _make_collector()
        snapshot = collector.collect_system_health()

        assert isinstance(snapshot, SystemHealthSnapshot)
        assert snapshot.total_equity == 50000.0
        assert snapshot.open_position_count == 2
        assert snapshot.current_drawdown == 0.03
        assert snapshot.stale_symbol_count == 0
        assert snapshot.total_symbols == 2

    def test_stale_symbols_counted(self) -> None:
        feed = _make_mock_feed()
        feed.stale_symbols = {"BTC/USDT"}
        collector = _make_collector(feed=feed)

        snapshot = collector.collect_system_health()
        assert snapshot.stale_symbol_count == 1

    def test_today_trades_filtered(self) -> None:
        analytics = _make_mock_analytics()
        trade = MagicMock()
        trade.exit_time = datetime.now(UTC)
        trade.pnl = 50.0
        analytics.closed_trades = [trade]
        collector = _make_collector(analytics=analytics)

        snapshot = collector.collect_system_health()
        assert snapshot.today_trades == 1
        assert snapshot.today_pnl == 50.0

    def test_old_trades_excluded(self) -> None:
        analytics = _make_mock_analytics()
        trade = MagicMock()
        trade.exit_time = datetime(2025, 1, 1, tzinfo=UTC)
        trade.pnl = 100.0
        analytics.closed_trades = [trade]
        collector = _make_collector(analytics=analytics)

        snapshot = collector.collect_system_health()
        assert snapshot.today_trades == 0
        assert snapshot.today_pnl == 0.0


# ─── Collect Strategy Health 테스트 ───────────────────────


class TestCollectStrategyHealth:
    def test_basic_snapshot(self) -> None:
        collector = _make_collector()
        snapshot = collector.collect_strategy_health()

        assert isinstance(snapshot, StrategyHealthSnapshot)
        assert snapshot.rolling_sharpe_30d == 0.0  # 거래 없음
        assert snapshot.total_closed_trades == 0

    def test_open_positions_collected(self) -> None:
        pm = _make_mock_pm()
        pos = MagicMock()
        pos.is_open = True
        pos.direction.name = "LONG"
        pos.unrealized_pnl = 120.0
        pos.size = 5.0
        pos.current_weight = 0.1
        pm.positions = {"SOL/USDT": pos}
        collector = _make_collector(pm=pm)

        snapshot = collector.collect_strategy_health()
        assert len(snapshot.open_positions) == 1
        assert snapshot.open_positions[0].symbol == "SOL/USDT"
        assert snapshot.open_positions[0].direction == "LONG"

    def test_alpha_decay_detection(self) -> None:
        collector = _make_collector()
        # CONFIRMATIONS=1 이므로 한 번만 하강이면 발동
        collector._sharpe_history = [2.0, 1.5, 1.0]
        # collect_strategy_health() 호출 시 0.0 추가 → [2.0, 1.5, 1.0, 0.0]
        # → last 3 = [1.0, 0.0, ...] → 실질적으로 detect에서 하강 감지
        snapshot = collector.collect_strategy_health()
        assert snapshot.alpha_decay_detected is True

    def test_no_alpha_decay_insufficient_history(self) -> None:
        collector = _make_collector()
        collector._sharpe_history = [1.0]  # 부족
        snapshot = collector.collect_strategy_health()
        assert snapshot.alpha_decay_detected is False


# ─── Rolling Sharpe 계산 테스트 ───────────────────────────


class TestComputeRollingSharpe:
    def test_empty_trades(self) -> None:
        assert HealthDataCollector.compute_rolling_sharpe([]) == 0.0

    def test_single_trade(self) -> None:
        trade = MagicMock()
        trade.pnl_pct = 5.0
        assert HealthDataCollector.compute_rolling_sharpe([trade]) == 0.0

    def test_positive_sharpe(self) -> None:
        trades = []
        for pnl in [1.0, 2.0, 1.5, 2.5, 1.0, 2.0, 1.5, 3.0]:
            t = MagicMock()
            t.pnl_pct = pnl
            trades.append(t)

        sharpe = HealthDataCollector.compute_rolling_sharpe(trades)
        assert sharpe > 0


# ─── Trade Stats 테스트 ──────────────────────────────────


class TestComputeTradeStats:
    def test_empty(self) -> None:
        wr, pf = HealthDataCollector.compute_trade_stats([])
        assert wr == 0.0
        assert pf == 0.0

    def test_all_winners(self) -> None:
        trades = [MagicMock(pnl=100.0), MagicMock(pnl=50.0)]
        wr, pf = HealthDataCollector.compute_trade_stats(trades)
        assert wr == 1.0
        assert pf == float("inf")

    def test_mixed_trades(self) -> None:
        trades = [MagicMock(pnl=100.0), MagicMock(pnl=-50.0)]
        wr, pf = HealthDataCollector.compute_trade_stats(trades)
        assert wr == 0.5
        assert pf == pytest.approx(2.0)


# ─── Alpha Decay Detection 테스트 ────────────────────────


class TestDetectAlphaDecay:
    def test_single_decline_triggered(self) -> None:
        """CONFIRMATIONS=1이므로 단일 하강 → 즉시 발동."""
        collector = _make_collector()
        collector._sharpe_history = [2.0, 1.5, 1.0]
        assert collector.detect_alpha_decay() is True
        assert collector._alpha_decay_streak == 1

    def test_recovery_resets_streak(self) -> None:
        """반등 시 streak 리셋."""
        collector = _make_collector()
        collector._sharpe_history = [2.0, 1.5, 1.0]
        collector.detect_alpha_decay()  # streak=1
        # 반등
        collector._sharpe_history = [1.5, 1.0, 1.5]
        assert collector.detect_alpha_decay() is False
        assert collector._alpha_decay_streak == 0

    def test_no_decline(self) -> None:
        collector = _make_collector()
        collector._sharpe_history = [1.0, 1.5, 2.0]
        assert collector.detect_alpha_decay() is False
        assert collector._alpha_decay_streak == 0

    def test_partial_decline(self) -> None:
        collector = _make_collector()
        collector._sharpe_history = [1.0, 2.0, 1.5]
        assert collector.detect_alpha_decay() is False

    def test_insufficient_history(self) -> None:
        collector = _make_collector()
        collector._sharpe_history = [1.0, 0.5]
        assert collector.detect_alpha_decay() is False
        assert collector._alpha_decay_streak == 0


# ─── Collect Regime Report 테스트 ────────────────────────


class TestCollectRegimeReport:
    @pytest.mark.asyncio
    async def test_returns_report(self) -> None:
        collector = _make_collector()

        sym_snap = SymbolDerivativesSnapshot(
            symbol="BTC/USDT",
            price=97420.0,
            funding_rate=0.0002,
            funding_rate_annualized=21.9,
            open_interest=5e9,
            ls_ratio=1.4,
            taker_ratio=1.1,
        )

        with patch.object(
            collector._snapshot_fetcher, "fetch_all", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = [sym_snap]
            report = await collector.collect_regime_report()

        assert report is not None
        assert report.regime_score != 0.0
        assert len(report.symbols) == 1

    @pytest.mark.asyncio
    async def test_no_data_returns_none(self) -> None:
        collector = _make_collector()

        with patch.object(
            collector._snapshot_fetcher, "fetch_all", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = []
            report = await collector.collect_regime_report()

        assert report is None


# ─── Lifecycle 테스트 ────────────────────────────────────


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        collector = _make_collector()
        with patch.object(collector._snapshot_fetcher, "start", new_callable=AsyncMock):
            await collector.start()

        with patch.object(collector._snapshot_fetcher, "stop", new_callable=AsyncMock):
            await collector.stop()
