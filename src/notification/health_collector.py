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
    BarCloseReportData,
    DailyReportData,
    PositionStatus,
    SignalChangeItem,
    StrategyHealthSnapshot,
    StrategyIndicatorItem,
    StrategyPerformanceSnapshot,
    SystemHealthSnapshot,
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
            aggregate_leverage=self._pm.aggregate_leverage,
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

    # ─── Daily Report Data Collection ─────────────────────────

    async def collect_daily_report_data(self) -> DailyReportData:
        """Spot Daily Report용 전체 데이터 수집 (5 sections)."""
        # Section 1: Strategy Info
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

        # Section 2: Portfolio Summary
        equity = self._pm.total_equity
        cash = self._pm.available_cash
        cash_pct = (cash / equity * 100) if equity > 0 else 0.0

        today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        trades = self._analytics.closed_trades
        trades_today = [t for t in trades if t.exit_time and t.exit_time >= today_start]
        today_pnl = sum(float(t.pnl) for t in trades_today if t.pnl is not None)

        initial = self._pm.initial_capital
        cum_return = ((equity - initial) / initial * 100) if initial > 0 else 0.0

        mdd = self._rm.current_drawdown * 100
        cutoff_30d = datetime.now(UTC) - timedelta(days=_ROLLING_SHARPE_DAYS)
        recent_30d = [t for t in trades if t.exit_time and t.exit_time >= cutoff_30d]
        sharpe = self.compute_rolling_sharpe(recent_30d)

        # Sharpe history + alpha decay (동일 로직 재사용)
        self._sharpe_history.append(sharpe)
        if len(self._sharpe_history) > _MAX_SHARPE_HISTORY:
            self._sharpe_history = self._sharpe_history[-_MAX_SHARPE_HISTORY:]
        alpha_decay = self.detect_alpha_decay()

        all_trades = list(trades)
        recent_n = all_trades[-_RECENT_TRADES_COUNT:] if all_trades else []
        win_rate, profit_factor = self.compute_trade_stats(recent_n)

        # Section 3: Asset Dashboard (async — fetch_ticker)
        assets = await self._collect_asset_dashboard()

        # Section 4: Strategy Indicators
        indicators = self._collect_strategy_indicators()

        # Section 5: System Health
        uptime = time.monotonic() - self._start_time
        ws_ok = len(self._symbols) - len(self._feed.stale_symbols)

        return DailyReportData(
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            trailing_stop_config=ts_config,
            timeframe=timeframe,
            total_equity=equity,
            available_cash=cash,
            cash_pct=cash_pct,
            today_pnl=today_pnl,
            invested_count=self._pm.open_position_count,
            total_asset_count=len(self._symbols),
            cumulative_return_pct=cum_return,
            max_drawdown_pct=mdd,
            rolling_sharpe_30d=sharpe,
            assets=tuple(assets),
            indicators=tuple(indicators),
            uptime_seconds=uptime,
            is_circuit_breaker_active=self._rm.is_circuit_breaker_active,
            ws_ok_count=ws_ok,
            ws_total_count=len(self._symbols),
            win_rate=win_rate,
            profit_factor=profit_factor,
            alpha_decay_detected=alpha_decay,
        )

    async def _collect_asset_dashboard(self) -> list[AssetDashboardItem]:
        """에셋별 가격/PnL/stop distance 수집."""
        if self._spot_client is None:
            return []

        positions = self._pm.positions
        active_stops: dict[str, Any] = (
            self._exchange_stop_mgr.active_stops
            if self._exchange_stop_mgr is not None
            else {}
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
            stop_dist = (
                (price - stop_price) / price * 100
                if stop_price and price > 0
                else None
            )

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

        return BarCloseReportData(
            bar_time_utc=bar_time_utc,
            signal_changes=tuple(signal_changes),
            assets=tuple(assets),
            total_equity=self._pm.total_equity,
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
