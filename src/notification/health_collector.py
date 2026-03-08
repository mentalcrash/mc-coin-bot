"""HealthDataCollector -- 시스템/전략/마켓 데이터 수집기 (스케줄링 없음).

HealthCheckScheduler에서 3개 asyncio 스케줄링 루프를 제거하고
데이터 수집 로직만 추출한 순수 수집기입니다.
ReportScheduler (Daily Report)에서 호출합니다.

Rules Applied:
    - EDA 패턴: asyncio lifecycle (start/stop)
    - #10 Python Standards: Async patterns, type hints
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.notification.health_models import (
    AssetDashboardItem,
    AssetMonthlyPerformance,
    AssetQuarterlyPerformance,
    AssetWeeklyPerformance,
    AssetYearlyPerformance,
    BarCloseReportData,
    DailyReportData,
    MonthlyPerformanceTrend,
    MonthlyReportData,
    PositionStatus,
    QuarterlyPerformanceTrend,
    QuarterlyReportData,
    SignalChangeItem,
    StrategyHealthSnapshot,
    StrategyIndicatorItem,
    StrategyPerformanceSnapshot,
    SystemHealthSnapshot,
    WeeklyReportData,
    YearlyReportData,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.event_bus import EventBus
    from src.eda.analytics import AnalyticsEngine
    from src.eda.live_data_feed import LiveDataFeed
    from src.eda.portfolio_manager import EDAPortfolioManager
    from src.eda.risk_manager import EDARiskManager
    from src.eda.strategy_engine import StrategyEngine
    from src.exchange.binance_spot_client import BinanceSpotClient
    from src.notification.queue import NotificationQueue

# Rolling Sharpe 계산 기간
_ROLLING_SHARPE_DAYS = 30
_RECENT_TRADES_COUNT = 20
_ALPHA_DECAY_WINDOW = 3
_ALPHA_DECAY_CONFIRMATIONS = 1  # 24h 주기이므로 1회 확인으로 충분
_MAX_SHARPE_HISTORY = 10
_SHARPE_HEALTHY_THRESHOLD = 0.5

# 연환산 계수 (일봉 기준)
_ANNUALIZE_SQRT = 365.0**0.5


class HealthDataCollector:
    """시스템/전략/마켓 데이터 수집기 (스케줄링 없음).

    Args:
        pm: EDAPortfolioManager
        rm: EDARiskManager
        analytics: AnalyticsEngine
        feed: LiveDataFeed
        bus: EventBus
        queue: NotificationQueue
        symbols: 모니터링 대상 심볼 리스트
        exchange_stop_mgr: ExchangeStopManager (선택)
    """

    def __init__(
        self,
        pm: EDAPortfolioManager,
        rm: EDARiskManager,
        analytics: AnalyticsEngine,
        feed: LiveDataFeed,
        bus: EventBus,
        queue: NotificationQueue,
        symbols: list[str],
        exchange_stop_mgr: Any = None,
        spot_client: BinanceSpotClient | None = None,
        strategy_engine: StrategyEngine | None = None,
    ) -> None:
        self._pm = pm
        self._rm = rm
        self._analytics = analytics
        self._feed = feed
        self._bus = bus
        self._queue = queue
        self._symbols = symbols
        self._exchange_stop_mgr = exchange_stop_mgr
        self._spot_client = spot_client
        self._strategy_engine = strategy_engine

        self._start_time = time.monotonic()
        self._sharpe_history: list[float] = []
        self._alpha_decay_streak: int = 0
        self._prev_signals: dict[str, str] = {}  # symbol → "LONG"/"NEUTRAL"

    async def start(self) -> None:
        """HealthDataCollector 시작."""
        logger.info("HealthDataCollector started")

    async def stop(self) -> None:
        """HealthDataCollector 정리."""
        logger.info("HealthDataCollector stopped")

    # ─── Public Collectors ─────────────────────────────────

    def collect_system_health(self) -> SystemHealthSnapshot:
        """PM/RM/Analytics/Feed/Bus에서 시스템 상태 수집."""
        uptime = time.monotonic() - self._start_time

        # 금일 거래 필터
        today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        trades = self._analytics.closed_trades
        trades_today = [t for t in trades if t.exit_time and t.exit_time >= today_start]
        today_pnl = sum(float(t.pnl) for t in trades_today if t.pnl is not None)

        # Safety stop 상태 수집
        safety_stop_count = 0
        safety_stop_failures = 0
        if self._exchange_stop_mgr is not None:
            active_stops = self._exchange_stop_mgr.active_stops
            safety_stop_count = len(active_stops)
            safety_stop_failures = max(
                (s.placement_failures for s in active_stops.values()), default=0
            )

        return SystemHealthSnapshot(
            timestamp=datetime.now(UTC),
            uptime_seconds=uptime,
            total_equity=self._pm.total_equity,
            available_cash=self._pm.available_cash,
            capital_utilization=self._pm.capital_utilization,
            open_position_count=self._pm.open_position_count,
            total_symbols=len(self._symbols),
            current_drawdown=self._rm.current_drawdown,
            peak_equity=self._rm.peak_equity,
            is_circuit_breaker_active=self._rm.is_circuit_breaker_active,
            today_pnl=today_pnl,
            today_trades=len(trades_today),
            stale_symbol_count=len(self._feed.stale_symbols),
            bars_emitted=self._feed.bars_emitted,
            events_dropped=self._bus.metrics.events_dropped,
            max_queue_depth=self._bus.metrics.max_queue_depth,
            is_notification_degraded=self._queue.is_degraded,
            safety_stop_count=safety_stop_count,
            safety_stop_failures=safety_stop_failures,
        )

    def collect_strategy_health(self) -> StrategyHealthSnapshot:
        """Analytics/PM/RM에서 전략 건강 상태 수집."""
        trades = self._analytics.closed_trades
        now = datetime.now(UTC)

        # Rolling Sharpe (30d)
        cutoff_30d = now - timedelta(days=_ROLLING_SHARPE_DAYS)
        recent_trades_30d = [t for t in trades if t.exit_time and t.exit_time >= cutoff_30d]
        rolling_sharpe = self.compute_rolling_sharpe(recent_trades_30d)

        # Alpha decay 감지
        self._sharpe_history.append(rolling_sharpe)
        if len(self._sharpe_history) > _MAX_SHARPE_HISTORY:
            self._sharpe_history = self._sharpe_history[-_MAX_SHARPE_HISTORY:]

        alpha_decay = self.detect_alpha_decay()

        # 최근 20건 win rate / profit factor
        trades_list = list(trades)
        recent_n = trades_list[-_RECENT_TRADES_COUNT:] if trades_list else []
        win_rate, profit_factor = self.compute_trade_stats(recent_n)

        # 오픈 포지션
        positions: list[PositionStatus] = []
        for sym, pos in self._pm.positions.items():
            if pos.is_open:
                positions.append(
                    PositionStatus(
                        symbol=sym,
                        direction=pos.direction.name,
                        unrealized_pnl=pos.unrealized_pnl,
                        size=pos.size,
                        current_weight=pos.current_weight,
                    )
                )

        # 전략별 breakdown (client_order_id에서 전략명 추출)
        strategy_breakdown = self.build_strategy_breakdown(recent_trades_30d)

        return StrategyHealthSnapshot(
            timestamp=now,
            rolling_sharpe_30d=rolling_sharpe,
            win_rate_recent=win_rate,
            profit_factor=profit_factor,
            total_closed_trades=len(trades),
            open_positions=tuple(positions),
            is_circuit_breaker_active=self._rm.is_circuit_breaker_active,
            alpha_decay_detected=alpha_decay,
            strategy_breakdown=tuple(strategy_breakdown),
        )

    # ─── Common Helpers ──────────────────────────────────────

    def _collect_strategy_info(self) -> dict[str, object]:
        """Strategy Info 공통 수집 (4필드)."""
        strategy = self._strategy_engine.strategy if self._strategy_engine else None
        strategy_name = strategy.name if strategy else "unknown"
        strategy_params: dict[str, str] = {}
        if strategy is not None and hasattr(strategy, "get_startup_info"):
            strategy_params = strategy.get_startup_info()

        ts_config = ""
        pm_cfg = self._pm.pm_config
        if pm_cfg.use_trailing_stop:
            ts_config = f"{pm_cfg.trailing_stop_atr_multiplier}x ATR"

        timeframe = self._strategy_engine.target_timeframe if self._strategy_engine else "?"

        return {
            "strategy_name": strategy_name,
            "strategy_params": strategy_params,
            "trailing_stop_config": ts_config,
            "timeframe": timeframe,
        }

    def _collect_system_health_fields(self) -> dict[str, object]:
        """System Health 공통 수집 (uptime, cb, ws, sharpe, win_rate, pf, alpha_decay)."""
        trades = self._analytics.closed_trades
        now = datetime.now(UTC)

        uptime = time.monotonic() - self._start_time
        ws_ok = len(self._symbols) - len(self._feed.stale_symbols)

        cutoff_30d = now - timedelta(days=_ROLLING_SHARPE_DAYS)
        recent_30d = [t for t in trades if t.exit_time and t.exit_time >= cutoff_30d]
        sharpe = self.compute_rolling_sharpe(recent_30d)

        self._sharpe_history.append(sharpe)
        if len(self._sharpe_history) > _MAX_SHARPE_HISTORY:
            self._sharpe_history = self._sharpe_history[-_MAX_SHARPE_HISTORY:]
        alpha_decay = self.detect_alpha_decay()

        all_trades = list(trades)
        recent_n = all_trades[-_RECENT_TRADES_COUNT:] if all_trades else []
        win_rate, profit_factor = self.compute_trade_stats(recent_n)

        return {
            "uptime_seconds": uptime,
            "is_circuit_breaker_active": self._rm.is_circuit_breaker_active,
            "ws_ok_count": ws_ok,
            "ws_total_count": len(self._symbols),
            "rolling_sharpe_30d": sharpe,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "alpha_decay_detected": alpha_decay,
        }

    def _collect_portfolio_summary(self) -> dict[str, object]:
        """Portfolio Summary 공통 수집 (equity, cash, cum_return, mdd 등)."""
        equity = self._pm.total_equity
        cash = self._pm.available_cash
        cash_pct = (cash / equity * 100) if equity > 0 else 0.0
        initial = self._pm.initial_capital
        cum_return = ((equity - initial) / initial * 100) if initial > 0 else 0.0
        mdd = self._rm.current_drawdown * 100

        return {
            "total_equity": equity,
            "available_cash": cash,
            "cash_pct": cash_pct,
            "invested_count": self._pm.open_position_count,
            "total_asset_count": len(self._symbols),
            "cumulative_return_pct": cum_return,
            "max_drawdown_pct": mdd,
        }

    def _extract_trade_stats(
        self,
        trades: Sequence[object],
    ) -> dict[str, object]:
        """기간별 trade summary (best/worst/win_rate/pf/avg_pnl/fees) 추출."""
        best_sym, best_pnl, worst_sym, worst_pnl = self._extract_best_worst(trades)
        wr, pf = self.compute_trade_stats(trades)
        pnls = [
            float(t.pnl)  # type: ignore[union-attr]
            for t in trades
            if getattr(t, "pnl", None) is not None
        ]
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0.0
        total_fees = sum(float(getattr(t, "fees", 0) or 0) for t in trades)
        return {
            "best_trade_symbol": best_sym,
            "best_trade_pnl": best_pnl,
            "worst_trade_symbol": worst_sym,
            "worst_trade_pnl": worst_pnl,
            "win_rate_period": wr,
            "profit_factor_period": pf,
            "avg_trade_pnl": avg_pnl,
            "total_fees": total_fees,
        }

    async def _collect_asset_period_performance(
        self,
        trades_period: Sequence[object],
    ) -> tuple[dict[str, float], dict[str, int], list[tuple[str, str, float, float]]]:
        """에셋별 기간 성과 공통 수집.

        Returns:
            (pnl_by_sym, count_by_sym, ticker_data: [(symbol, signal, price, change_pct)])
        """
        pnl_by_sym: dict[str, float] = {}
        count_by_sym: dict[str, int] = {}
        for t in trades_period:
            sym = getattr(t, "symbol", None)
            pnl = getattr(t, "pnl", None)
            if sym and pnl is not None:
                pnl_by_sym[sym] = pnl_by_sym.get(sym, 0.0) + float(pnl)
                count_by_sym[sym] = count_by_sym.get(sym, 0) + 1

        positions = self._pm.positions
        ticker_data: list[tuple[str, str, float, float]] = []
        if self._spot_client is not None:
            for symbol in self._symbols:
                try:
                    ticker = await self._spot_client.fetch_ticker(symbol)
                except Exception:
                    logger.warning("fetch_ticker failed for {}, skipping", symbol)
                    continue

                price = float(ticker.get("last", 0) or 0)
                change_pct = float(ticker.get("percentage", 0) or 0)
                pos = positions.get(symbol)
                is_open = pos is not None and pos.is_open
                signal = pos.direction.name if pos and is_open else "NEUTRAL"
                ticker_data.append((symbol, signal, price, change_pct))

        return pnl_by_sym, count_by_sym, ticker_data

    # ─── Daily Report Data Collection ─────────────────────────

    async def collect_daily_report_data(self) -> DailyReportData:
        """Spot Daily Report용 전체 데이터 수집 (5 sections)."""
        si = self._collect_strategy_info()
        ps = self._collect_portfolio_summary()

        today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        trades = self._analytics.closed_trades
        trades_today = [t for t in trades if t.exit_time and t.exit_time >= today_start]
        today_pnl = sum(float(t.pnl) for t in trades_today if t.pnl is not None)

        sh = self._collect_system_health_fields()
        assets = await self._collect_asset_dashboard()
        indicators = self._collect_strategy_indicators()

        return DailyReportData(
            **si,  # type: ignore[arg-type]
            total_equity=ps["total_equity"],  # type: ignore[arg-type]
            available_cash=ps["available_cash"],  # type: ignore[arg-type]
            cash_pct=ps["cash_pct"],  # type: ignore[arg-type]
            today_pnl=today_pnl,
            invested_count=ps["invested_count"],  # type: ignore[arg-type]
            total_asset_count=ps["total_asset_count"],  # type: ignore[arg-type]
            cumulative_return_pct=ps["cumulative_return_pct"],  # type: ignore[arg-type]
            max_drawdown_pct=ps["max_drawdown_pct"],  # type: ignore[arg-type]
            rolling_sharpe_30d=sh["rolling_sharpe_30d"],  # type: ignore[arg-type]
            assets=tuple(assets),
            indicators=tuple(indicators),
            uptime_seconds=sh["uptime_seconds"],  # type: ignore[arg-type]
            is_circuit_breaker_active=sh["is_circuit_breaker_active"],  # type: ignore[arg-type]
            ws_ok_count=sh["ws_ok_count"],  # type: ignore[arg-type]
            ws_total_count=sh["ws_total_count"],  # type: ignore[arg-type]
            win_rate=sh["win_rate"],  # type: ignore[arg-type]
            profit_factor=sh["profit_factor"],  # type: ignore[arg-type]
            alpha_decay_detected=sh["alpha_decay_detected"],  # type: ignore[arg-type]
        )

    async def _collect_asset_dashboard(self) -> list[AssetDashboardItem]:
        """에셋별 가격/PnL/stop distance 수집."""
        if self._spot_client is None:
            return []

        positions = self._pm.positions
        active_stops: dict[str, Any] = (
            self._exchange_stop_mgr.active_stops if self._exchange_stop_mgr is not None else {}
        )

        items: list[AssetDashboardItem] = []
        for symbol in self._symbols:
            try:
                ticker = await self._spot_client.fetch_ticker(symbol)
            except Exception:
                logger.warning("fetch_ticker failed for {}, skipping", symbol)
                continue

            price = float(ticker.get("last", 0) or 0)
            change_24h = float(ticker.get("percentage", 0) or 0)

            pos = positions.get(symbol)
            is_open = pos is not None and pos.is_open
            pos_value = pos.size * pos.last_price if pos and is_open else 0.0
            day_pnl = pos.unrealized_pnl if pos and is_open else 0.0
            signal = pos.direction.name if pos and is_open else "NEUTRAL"

            stop_state = active_stops.get(symbol)
            stop_price = stop_state.stop_price if stop_state else None
            stop_dist = (price - stop_price) / price * 100 if stop_price and price > 0 else None

            items.append(
                AssetDashboardItem(
                    symbol=symbol,
                    signal=signal,
                    current_price=price,
                    change_24h_pct=change_24h,
                    position_value=pos_value,
                    day_pnl=day_pnl,
                    stop_price=stop_price,
                    stop_distance_pct=stop_dist,
                )
            )
        return items

    def _collect_strategy_indicators(self) -> list[StrategyIndicatorItem]:
        """StrategyEngine 캐시에서 에셋별 indicator 수집."""
        if self._strategy_engine is None:
            return []

        indicators = self._strategy_engine.latest_indicators
        items: list[StrategyIndicatorItem] = []

        for symbol in self._symbols:
            ind = indicators.get(symbol, {})
            st_line = ind.get("supertrend")
            adx_val = ind.get("adx")
            st_dir = ind.get("supertrend_dir")
            outlook = self._determine_outlook(adx_val, st_dir)

            items.append(
                StrategyIndicatorItem(
                    symbol=symbol,
                    supertrend_line=st_line,
                    adx_value=adx_val,
                    outlook=outlook,
                )
            )
        return items

    _OUTLOOK_ADX_THRESHOLD = 25
    _OUTLOOK_ADX_STRONG = 35

    @staticmethod
    def _determine_outlook(adx: float | None, st_dir: float | None) -> str:
        """ADX + SuperTrend direction → outlook 텍스트."""
        if adx is None:
            return "데이터 없음"
        is_long = st_dir is not None and st_dir > 0
        if adx < HealthDataCollector._OUTLOOK_ADX_THRESHOLD:
            return "비추세"
        if is_long:
            return "강한 상승추세" if adx > HealthDataCollector._OUTLOOK_ADX_STRONG else "상승추세"
        return "하락 전환 대기"

    # ─── Weekly Report Data Collection ──────────────────────────

    async def collect_weekly_report_data(self) -> WeeklyReportData:
        """Spot Weekly Report용 전체 데이터 수집 (6 sections)."""
        now = datetime.now(UTC)
        week_start = (now - timedelta(days=now.weekday())).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        si = self._collect_strategy_info()
        ps = self._collect_portfolio_summary()

        trades = self._analytics.closed_trades
        trades_week = [t for t in trades if t.exit_time and t.exit_time >= week_start]
        week_pnl = sum(float(t.pnl) for t in trades_week if t.pnl is not None)

        asset_perfs = await self._collect_asset_weekly_performance(trades_week)
        ts = self._extract_trade_stats(trades_week)
        indicators = self._collect_strategy_indicators()
        sh = self._collect_system_health_fields()

        return WeeklyReportData(
            **si,  # type: ignore[arg-type]
            **ps,  # type: ignore[arg-type]
            **sh,  # type: ignore[arg-type]
            week_pnl=week_pnl,
            week_trades=len(trades_week),
            assets=tuple(asset_perfs),
            best_trade_symbol=ts["best_trade_symbol"],  # type: ignore[arg-type]
            best_trade_pnl=ts["best_trade_pnl"],  # type: ignore[arg-type]
            worst_trade_symbol=ts["worst_trade_symbol"],  # type: ignore[arg-type]
            worst_trade_pnl=ts["worst_trade_pnl"],  # type: ignore[arg-type]
            week_win_rate=ts["win_rate_period"],  # type: ignore[arg-type]
            week_profit_factor=ts["profit_factor_period"],  # type: ignore[arg-type]
            indicators=tuple(indicators),
        )

    async def _collect_asset_weekly_performance(
        self,
        trades_week: Sequence[object],
    ) -> list[AssetWeeklyPerformance]:
        """에셋별 주간 성과 수집."""
        if self._spot_client is None:
            return []

        # 에셋별 주간 거래 집계
        pnl_by_sym: dict[str, float] = {}
        count_by_sym: dict[str, int] = {}
        for t in trades_week:
            sym = getattr(t, "symbol", None)
            pnl = getattr(t, "pnl", None)
            if sym and pnl is not None:
                pnl_by_sym[sym] = pnl_by_sym.get(sym, 0.0) + float(pnl)
                count_by_sym[sym] = count_by_sym.get(sym, 0) + 1

        positions = self._pm.positions
        items: list[AssetWeeklyPerformance] = []
        for symbol in self._symbols:
            try:
                ticker = await self._spot_client.fetch_ticker(symbol)
            except Exception:
                logger.warning("fetch_ticker failed for {}, skipping", symbol)
                continue

            price = float(ticker.get("last", 0) or 0)
            change_pct = float(ticker.get("percentage", 0) or 0)

            pos = positions.get(symbol)
            is_open = pos is not None and pos.is_open
            signal = pos.direction.name if pos and is_open else "NEUTRAL"

            items.append(
                AssetWeeklyPerformance(
                    symbol=symbol,
                    signal=signal,
                    current_price=price,
                    week_change_pct=change_pct,
                    week_pnl=pnl_by_sym.get(symbol, 0.0),
                    week_trades=count_by_sym.get(symbol, 0),
                )
            )
        return items

    @staticmethod
    def _extract_best_worst(
        trades: Sequence[object],
    ) -> tuple[str, float, str, float]:
        """주간 거래에서 best/worst trade 추출.

        Returns:
            (best_symbol, best_pnl, worst_symbol, worst_pnl)
        """
        if not trades:
            return "—", 0.0, "—", 0.0

        best_t = max(
            trades,
            key=lambda t: float(getattr(t, "pnl", 0) or 0),
        )
        worst_t = min(
            trades,
            key=lambda t: float(getattr(t, "pnl", 0) or 0),
        )
        return (
            getattr(best_t, "symbol", "?"),
            float(getattr(best_t, "pnl", 0) or 0),
            getattr(worst_t, "symbol", "?"),
            float(getattr(worst_t, "pnl", 0) or 0),
        )

    # ─── Monthly Report Data Collection ─────────────────────────

    _PERFORMANCE_TREND_MONTHS = 3

    async def collect_monthly_report_data(self) -> MonthlyReportData:
        """Spot Monthly Report용 전체 데이터 수집 (8 sections)."""
        now = datetime.now(UTC)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        si = self._collect_strategy_info()
        ps = self._collect_portfolio_summary()
        equity = ps["total_equity"]

        trades = self._analytics.closed_trades
        trades_month = [t for t in trades if t.exit_time and t.exit_time >= month_start]
        month_pnl = sum(float(t.pnl) for t in trades_month if t.pnl is not None)

        equity_at_month_start = float(equity) - month_pnl  # type: ignore[arg-type]
        month_return = (
            (month_pnl / equity_at_month_start * 100) if equity_at_month_start > 0 else 0.0
        )

        asset_perfs = await self._collect_asset_monthly_performance(trades_month)
        ts = self._extract_trade_stats(trades_month)
        trend = self._build_performance_trend(list(trades))
        indicators = self._collect_strategy_indicators()
        sh = self._collect_system_health_fields()

        month_max_dd = self._compute_month_max_drawdown(trades_month, float(equity))  # type: ignore[arg-type]
        longest_streak = self._compute_longest_losing_streak(trades_month)

        return MonthlyReportData(
            **si,  # type: ignore[arg-type]
            **ps,  # type: ignore[arg-type]
            **sh,  # type: ignore[arg-type]
            month_pnl=month_pnl,
            month_trades=len(trades_month),
            month_return_pct=month_return,
            assets=tuple(asset_perfs),
            best_trade_symbol=ts["best_trade_symbol"],  # type: ignore[arg-type]
            best_trade_pnl=ts["best_trade_pnl"],  # type: ignore[arg-type]
            worst_trade_symbol=ts["worst_trade_symbol"],  # type: ignore[arg-type]
            worst_trade_pnl=ts["worst_trade_pnl"],  # type: ignore[arg-type]
            month_win_rate=ts["win_rate_period"],  # type: ignore[arg-type]
            month_profit_factor=ts["profit_factor_period"],  # type: ignore[arg-type]
            avg_trade_pnl=ts["avg_trade_pnl"],  # type: ignore[arg-type]
            total_fees=ts["total_fees"],  # type: ignore[arg-type]
            performance_trend=tuple(trend),
            indicators=tuple(indicators),
            month_max_drawdown_pct=month_max_dd,
            longest_losing_streak=longest_streak,
        )

    async def _collect_asset_monthly_performance(
        self,
        trades_month: Sequence[object],
    ) -> list[AssetMonthlyPerformance]:
        """에셋별 월간 성과 수집."""
        if self._spot_client is None:
            return []

        pnl_by_sym: dict[str, float] = {}
        count_by_sym: dict[str, int] = {}
        for t in trades_month:
            sym = getattr(t, "symbol", None)
            pnl = getattr(t, "pnl", None)
            if sym and pnl is not None:
                pnl_by_sym[sym] = pnl_by_sym.get(sym, 0.0) + float(pnl)
                count_by_sym[sym] = count_by_sym.get(sym, 0) + 1

        positions = self._pm.positions
        items: list[AssetMonthlyPerformance] = []
        for symbol in self._symbols:
            try:
                ticker = await self._spot_client.fetch_ticker(symbol)
            except Exception:
                logger.warning("fetch_ticker failed for {}, skipping", symbol)
                continue

            price = float(ticker.get("last", 0) or 0)
            change_pct = float(ticker.get("percentage", 0) or 0)

            pos = positions.get(symbol)
            is_open = pos is not None and pos.is_open
            signal = pos.direction.name if pos and is_open else "NEUTRAL"

            items.append(
                AssetMonthlyPerformance(
                    symbol=symbol,
                    signal=signal,
                    current_price=price,
                    month_change_pct=change_pct,
                    month_pnl=pnl_by_sym.get(symbol, 0.0),
                    month_trades=count_by_sym.get(symbol, 0),
                )
            )
        return items

    def _build_performance_trend(
        self,
        all_trades: list[object],
        months: int | None = None,
    ) -> list[MonthlyPerformanceTrend]:
        """최근 N개월 월별 성과 추이 계산."""
        num_months = months if months is not None else self._PERFORMANCE_TREND_MONTHS

        now = datetime.now(UTC)
        result: list[MonthlyPerformanceTrend] = []

        for i in range(num_months):
            # i=0: 이번 달, i=1: 지난 달, ...
            ref = now.replace(day=1) - timedelta(days=i * 28)
            ref = ref.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            year_month = ref.strftime("%Y-%m")

            # 해당 월 다음 달 첫째 날
            if ref.month == 12:  # noqa: PLR2004
                next_month = ref.replace(year=ref.year + 1, month=1)
            else:
                next_month = ref.replace(month=ref.month + 1)

            month_trades = [
                t
                for t in all_trades
                if getattr(t, "exit_time", None) and ref <= t.exit_time < next_month  # type: ignore[operator]
            ]

            if not month_trades:
                continue

            pnl = sum(
                float(t.pnl)  # type: ignore[union-attr]
                for t in month_trades
                if getattr(t, "pnl", None) is not None
            )
            sharpe = self.compute_rolling_sharpe(month_trades)

            # 월 수익률: 누적 PnL / 현재 equity에서 역산 (근사치)
            equity = self._pm.total_equity
            return_pct = (pnl / equity * 100) if equity > 0 else 0.0

            result.append(
                MonthlyPerformanceTrend(
                    year_month=year_month,
                    pnl=pnl,
                    return_pct=return_pct,
                    trades=len(month_trades),
                    sharpe=sharpe,
                )
            )

        return result

    @staticmethod
    def _compute_month_max_drawdown(
        trades_month: Sequence[object],
        current_equity: float,
    ) -> float:
        """월간 거래 기반 최대 낙폭 (%) 근사 계산."""
        if not trades_month:
            return 0.0

        pnls = [
            float(t.pnl)  # type: ignore[union-attr]
            for t in trades_month
            if getattr(t, "pnl", None) is not None
        ]
        if not pnls:
            return 0.0

        # 누적 equity curve → max drawdown
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cumulative += p
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        return (max_dd / current_equity * 100) if current_equity > 0 else 0.0

    @staticmethod
    def _compute_longest_losing_streak(trades: Sequence[object]) -> int:
        """최장 연패 수 계산."""
        max_streak = 0
        current_streak = 0
        for t in trades:
            pnl = getattr(t, "pnl", None)
            if pnl is not None and float(pnl) < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    # ─── Quarterly Report Data Collection ─────────────────────────

    async def collect_quarterly_report_data(self) -> QuarterlyReportData:
        """Spot Quarterly Report용 전체 데이터 수집 (8 sections)."""
        now = datetime.now(UTC)
        quarter_month = ((now.month - 1) // 3) * 3 + 1
        quarter_start = now.replace(
            month=quarter_month, day=1, hour=0, minute=0, second=0, microsecond=0
        )

        si = self._collect_strategy_info()
        ps = self._collect_portfolio_summary()
        equity = ps["total_equity"]

        trades = self._analytics.closed_trades
        trades_quarter = [t for t in trades if t.exit_time and t.exit_time >= quarter_start]
        quarter_pnl = sum(float(t.pnl) for t in trades_quarter if t.pnl is not None)

        equity_at_start = float(equity) - quarter_pnl  # type: ignore[arg-type]
        quarter_return = (quarter_pnl / equity_at_start * 100) if equity_at_start > 0 else 0.0

        asset_perfs = await self._collect_asset_quarterly_performance(trades_quarter)
        ts = self._extract_trade_stats(trades_quarter)
        trend = self._build_performance_trend(list(trades))
        indicators = self._collect_strategy_indicators()
        sh = self._collect_system_health_fields()

        q_max_dd = self._compute_month_max_drawdown(trades_quarter, float(equity))  # type: ignore[arg-type]
        longest_streak = self._compute_longest_losing_streak(trades_quarter)

        return QuarterlyReportData(
            **si,  # type: ignore[arg-type]
            **ps,  # type: ignore[arg-type]
            **sh,  # type: ignore[arg-type]
            quarter_pnl=quarter_pnl,
            quarter_trades=len(trades_quarter),
            quarter_return_pct=quarter_return,
            assets=tuple(asset_perfs),
            best_trade_symbol=ts["best_trade_symbol"],  # type: ignore[arg-type]
            best_trade_pnl=ts["best_trade_pnl"],  # type: ignore[arg-type]
            worst_trade_symbol=ts["worst_trade_symbol"],  # type: ignore[arg-type]
            worst_trade_pnl=ts["worst_trade_pnl"],  # type: ignore[arg-type]
            quarter_win_rate=ts["win_rate_period"],  # type: ignore[arg-type]
            quarter_profit_factor=ts["profit_factor_period"],  # type: ignore[arg-type]
            avg_trade_pnl=ts["avg_trade_pnl"],  # type: ignore[arg-type]
            total_fees=ts["total_fees"],  # type: ignore[arg-type]
            performance_trend=tuple(trend),
            indicators=tuple(indicators),
            quarter_max_drawdown_pct=q_max_dd,
            longest_losing_streak=longest_streak,
        )

    async def _collect_asset_quarterly_performance(
        self,
        trades_quarter: Sequence[object],
    ) -> list[AssetQuarterlyPerformance]:
        """에셋별 분기 성과 수집."""
        pnl_by_sym, count_by_sym, ticker_data = await self._collect_asset_period_performance(
            trades_quarter
        )
        return [
            AssetQuarterlyPerformance(
                symbol=sym,
                signal=sig,
                current_price=price,
                quarter_change_pct=change,
                quarter_pnl=pnl_by_sym.get(sym, 0.0),
                quarter_trades=count_by_sym.get(sym, 0),
            )
            for sym, sig, price, change in ticker_data
        ]

    # ─── Yearly Report Data Collection ──────────────────────────

    async def collect_yearly_report_data(self) -> YearlyReportData:
        """Spot Yearly Report용 전체 데이터 수집 (9 sections)."""
        now = datetime.now(UTC)
        year_start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

        si = self._collect_strategy_info()
        ps = self._collect_portfolio_summary()
        equity = ps["total_equity"]

        trades = self._analytics.closed_trades
        trades_year = [t for t in trades if t.exit_time and t.exit_time >= year_start]
        year_pnl = sum(float(t.pnl) for t in trades_year if t.pnl is not None)

        equity_at_start = float(equity) - year_pnl  # type: ignore[arg-type]
        year_return = (year_pnl / equity_at_start * 100) if equity_at_start > 0 else 0.0

        asset_perfs = await self._collect_asset_yearly_performance(trades_year)
        ts = self._extract_trade_stats(trades_year)
        q_trend = self._build_quarterly_performance_trend(list(trades))
        m_trend = self._build_performance_trend(list(trades), months=12)
        indicators = self._collect_strategy_indicators()
        sh = self._collect_system_health_fields()

        y_max_dd = self._compute_month_max_drawdown(trades_year, float(equity))  # type: ignore[arg-type]
        longest_streak = self._compute_longest_losing_streak(trades_year)

        return YearlyReportData(
            **si,  # type: ignore[arg-type]
            **ps,  # type: ignore[arg-type]
            **sh,  # type: ignore[arg-type]
            year_pnl=year_pnl,
            year_trades=len(trades_year),
            year_return_pct=year_return,
            assets=tuple(asset_perfs),
            best_trade_symbol=ts["best_trade_symbol"],  # type: ignore[arg-type]
            best_trade_pnl=ts["best_trade_pnl"],  # type: ignore[arg-type]
            worst_trade_symbol=ts["worst_trade_symbol"],  # type: ignore[arg-type]
            worst_trade_pnl=ts["worst_trade_pnl"],  # type: ignore[arg-type]
            year_win_rate=ts["win_rate_period"],  # type: ignore[arg-type]
            year_profit_factor=ts["profit_factor_period"],  # type: ignore[arg-type]
            avg_trade_pnl=ts["avg_trade_pnl"],  # type: ignore[arg-type]
            total_fees=ts["total_fees"],  # type: ignore[arg-type]
            quarterly_trend=tuple(q_trend),
            monthly_trend=tuple(m_trend),
            indicators=tuple(indicators),
            year_max_drawdown_pct=y_max_dd,
            longest_losing_streak=longest_streak,
        )

    async def _collect_asset_yearly_performance(
        self,
        trades_year: Sequence[object],
    ) -> list[AssetYearlyPerformance]:
        """에셋별 연간 성과 수집."""
        pnl_by_sym, count_by_sym, ticker_data = await self._collect_asset_period_performance(
            trades_year
        )
        return [
            AssetYearlyPerformance(
                symbol=sym,
                signal=sig,
                current_price=price,
                year_change_pct=change,
                year_pnl=pnl_by_sym.get(sym, 0.0),
                year_trades=count_by_sym.get(sym, 0),
            )
            for sym, sig, price, change in ticker_data
        ]

    def _build_quarterly_performance_trend(
        self,
        all_trades: list[object],
    ) -> list[QuarterlyPerformanceTrend]:
        """최근 4분기 성과 추이 계산."""
        now = datetime.now(UTC)
        quarters: list[QuarterlyPerformanceTrend] = []

        for i in range(4):
            # i=0: 현재 분기, i=1: 지난 분기, ...
            ref = now - timedelta(days=i * 91)
            q_month = ((ref.month - 1) // 3) * 3 + 1
            q_start = ref.replace(month=q_month, day=1, hour=0, minute=0, second=0, microsecond=0)
            q_num = (q_month - 1) // 3 + 1
            label = f"{q_start.year}-Q{q_num}"

            # 분기 끝
            if q_month + 3 > 12:  # noqa: PLR2004
                q_end = q_start.replace(year=q_start.year + 1, month=(q_month + 3) - 12)
            else:
                q_end = q_start.replace(month=q_month + 3)

            q_trades = [
                t
                for t in all_trades
                if getattr(t, "exit_time", None) and q_start <= t.exit_time < q_end  # type: ignore[operator]
            ]

            if not q_trades:
                continue

            pnl = sum(
                float(t.pnl)  # type: ignore[union-attr]
                for t in q_trades
                if getattr(t, "pnl", None) is not None
            )
            sharpe = self.compute_rolling_sharpe(q_trades)
            equity = self._pm.total_equity
            return_pct = (pnl / equity * 100) if equity > 0 else 0.0

            quarters.append(
                QuarterlyPerformanceTrend(
                    year_quarter=label,
                    pnl=pnl,
                    return_pct=return_pct,
                    trades=len(q_trades),
                    sharpe=sharpe,
                )
            )

        return quarters

    # ─── Bar Close Report ──────────────────────────────────────

    async def collect_bar_close_report_data(self, bar_time_utc: str) -> BarCloseReportData:
        """12H Bar Close Report 데이터 수집.

        Args:
            bar_time_utc: 봉 마감 시각 ("00:00" / "12:00")
        """
        # Section 1: Signal Changes
        signal_changes = self._detect_signal_changes()

        # Section 2: Asset Dashboard
        assets = await self._collect_asset_dashboard()

        # Section 3: Portfolio Snapshot
        today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        trades = self._analytics.closed_trades
        trades_today = [t for t in trades if t.exit_time and t.exit_time >= today_start]
        today_pnl = sum(float(t.pnl) for t in trades_today if t.pnl is not None)

        # Section 4: System Status
        uptime = time.monotonic() - self._start_time
        ws_ok = len(self._symbols) - len(self._feed.stale_symbols)

        equity = self._pm.total_equity
        cash = self._pm.available_cash
        deployed = equity - cash
        dd_pct = self._rm.current_drawdown
        util = self._pm.capital_utilization

        return BarCloseReportData(
            bar_time_utc=bar_time_utc,
            signal_changes=tuple(signal_changes),
            assets=tuple(assets),
            total_equity=equity,
            available_cash=cash,
            capital_deployed=deployed,
            drawdown_pct=dd_pct,
            capital_utilization=util,
            today_pnl=today_pnl,
            invested_count=self._pm.open_position_count,
            total_asset_count=len(self._symbols),
            uptime_seconds=uptime,
            is_circuit_breaker_active=self._rm.is_circuit_breaker_active,
            ws_ok_count=ws_ok,
            ws_total_count=len(self._symbols),
        )

    def _detect_signal_changes(self) -> list[SignalChangeItem]:
        """이전 스냅샷 대비 포지션 방향 변동 감지 + 스냅샷 갱신."""
        positions = self._pm.positions
        changes: list[SignalChangeItem] = []

        # 현재 상태 수집
        current: dict[str, str] = {}
        for symbol in self._symbols:
            pos = positions.get(symbol)
            current[symbol] = pos.direction.name if pos and pos.is_open else "NEUTRAL"

        # 이전 상태와 비교
        for symbol in self._symbols:
            prev = self._prev_signals.get(symbol, "NEUTRAL")
            curr = current[symbol]
            if prev != curr:
                # 청산 시 realized PnL 계산
                realized_pnl: float | None = None
                if curr == "NEUTRAL":
                    pos = positions.get(symbol)
                    if pos is not None:
                        realized_pnl = pos.realized_pnl

                changes.append(
                    SignalChangeItem(
                        symbol=symbol,
                        prev_signal=prev,
                        new_signal=curr,
                        realized_pnl=realized_pnl,
                    )
                )

        # 스냅샷 갱신
        self._prev_signals = current
        return changes

    # ─── Helper Functions ─────────────────────────────────────

    def build_strategy_breakdown(
        self,
        trades_30d: Sequence[object],
    ) -> list[StrategyPerformanceSnapshot]:
        """client_order_id에서 전략별로 그룹핑하여 성과 스냅샷 생성.

        Args:
            trades_30d: 최근 30일 TradeRecord 리스트

        Returns:
            전략별 성과 스냅샷 리스트
        """
        from collections import defaultdict

        # 전략별 그룹핑
        by_strategy: dict[str, list[object]] = defaultdict(list)
        for t in trades_30d:
            oid = getattr(t, "client_order_id", None) or ""
            parts = str(oid).rsplit("-", 2)
            name = parts[0] if len(parts) >= 3 else "unknown"  # noqa: PLR2004
            by_strategy[name].append(t)

        result: list[StrategyPerformanceSnapshot] = []
        for strategy_name, strades in sorted(by_strategy.items()):
            sharpe = self.compute_rolling_sharpe(strades)
            win_rate, _ = self.compute_trade_stats(strades)
            total_pnl = sum(
                float(t.pnl)  # type: ignore[union-attr]
                for t in strades
                if getattr(t, "pnl", None) is not None
            )

            # 상태 아이콘 결정
            if sharpe > _SHARPE_HEALTHY_THRESHOLD:
                status = "HEALTHY"
            elif sharpe >= 0:
                status = "WATCH"
            else:
                status = "DEGRADING"

            result.append(
                StrategyPerformanceSnapshot(
                    strategy_name=strategy_name,
                    rolling_sharpe=sharpe,
                    win_rate=win_rate,
                    total_pnl=total_pnl,
                    trade_count=len(strades),
                    status=status,
                )
            )

        return result

    @staticmethod
    def compute_rolling_sharpe(
        trades: Sequence[object],
    ) -> float:
        """거래 리스트에서 rolling Sharpe 계산.

        Args:
            trades: TradeRecord 리스트 (pnl_pct 속성 필요)

        Returns:
            Sharpe ratio (거래 없으면 0.0)
        """
        pnl_pcts = [
            float(t.pnl_pct)  # type: ignore[union-attr]
            for t in trades
            if getattr(t, "pnl_pct", None) is not None
        ]
        if len(pnl_pcts) < 2:  # noqa: PLR2004
            return 0.0

        mean = sum(pnl_pcts) / len(pnl_pcts)
        variance = sum((x - mean) ** 2 for x in pnl_pcts) / (len(pnl_pcts) - 1)
        std = variance**0.5
        if std == 0:
            return 0.0

        return (mean / std) * _ANNUALIZE_SQRT

    def detect_alpha_decay(self) -> bool:
        """직전 3개 Sharpe가 연속 하강 추세인지 감지.

        24h 주기 (Daily Report) x 1회 확인 = 즉시 발동.
        3일 연속 Sharpe 하강이면 감지.
        """
        if len(self._sharpe_history) < _ALPHA_DECAY_WINDOW:
            self._alpha_decay_streak = 0
            return False

        recent = self._sharpe_history[-_ALPHA_DECAY_WINDOW:]
        is_declining = all(recent[i] > recent[i + 1] for i in range(_ALPHA_DECAY_WINDOW - 1))

        if is_declining:
            self._alpha_decay_streak += 1
        else:
            self._alpha_decay_streak = 0

        return self._alpha_decay_streak >= _ALPHA_DECAY_CONFIRMATIONS

    @staticmethod
    def compute_trade_stats(
        trades: Sequence[object],
    ) -> tuple[float, float]:
        """거래 리스트에서 win rate과 profit factor 계산.

        Args:
            trades: TradeRecord 리스트

        Returns:
            (win_rate, profit_factor) 튜플
        """
        if not trades:
            return 0.0, 0.0

        pnls = [
            float(t.pnl)  # type: ignore[union-attr]
            for t in trades
            if getattr(t, "pnl", None) is not None
        ]
        if not pnls:
            return 0.0, 0.0

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls)

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return win_rate, profit_factor
