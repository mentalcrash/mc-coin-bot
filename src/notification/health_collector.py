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

from src.data.derivatives_snapshot import DerivativesSnapshotFetcher
from src.data.regime_score import classify_regime, compute_regime_score
from src.notification.health_models import (
    MarketRegimeReport,
    PositionStatus,
    StrategyHealthSnapshot,
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
    from src.exchange.binance_futures_client import BinanceFuturesClient
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
        futures_client: BinanceFuturesClient (None이면 내부 ccxt 생성)
        symbols: 모니터링 대상 심볼 리스트
        exchange_stop_mgr: ExchangeStopManager (선택)
        onchain_feed: LiveOnchainFeed (선택)
    """

    def __init__(
        self,
        pm: EDAPortfolioManager,
        rm: EDARiskManager,
        analytics: AnalyticsEngine,
        feed: LiveDataFeed,
        bus: EventBus,
        queue: NotificationQueue,
        futures_client: BinanceFuturesClient | None,
        symbols: list[str],
        exchange_stop_mgr: Any = None,
        onchain_feed: Any = None,
    ) -> None:
        self._pm = pm
        self._rm = rm
        self._analytics = analytics
        self._feed = feed
        self._bus = bus
        self._queue = queue
        self._symbols = symbols
        self._exchange_stop_mgr = exchange_stop_mgr
        self._onchain_feed = onchain_feed
        self._snapshot_fetcher = DerivativesSnapshotFetcher(futures_client)

        self._start_time = time.monotonic()
        self._sharpe_history: list[float] = []
        self._alpha_decay_streak: int = 0

    async def start(self) -> None:
        """DerivativesSnapshotFetcher 시작."""
        await self._snapshot_fetcher.start()
        logger.info("HealthDataCollector started")

    async def stop(self) -> None:
        """DerivativesSnapshotFetcher 정리."""
        await self._snapshot_fetcher.stop()
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

        # On-chain 상태 수집
        onchain_sources_ok = 0
        onchain_sources_total = 0
        onchain_cache_columns = 0
        if self._onchain_feed is not None:
            health = self._onchain_feed.get_health_status()
            onchain_cache_columns = health["total_columns"]
            onchain_sources_total, onchain_sources_ok = self._count_onchain_sources()

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
            onchain_sources_ok=onchain_sources_ok,
            onchain_sources_total=onchain_sources_total,
            onchain_cache_columns=onchain_cache_columns,
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
        recent_n = trades[-_RECENT_TRADES_COUNT:] if trades else []
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

    async def collect_regime_report(self) -> MarketRegimeReport | None:
        """Market Regime 데이터 수집.

        Returns:
            MarketRegimeReport 또는 None (데이터 없을 때)
        """
        symbol_snapshots = await self._snapshot_fetcher.fetch_all(self._symbols)

        if not symbol_snapshots:
            logger.warning("No derivatives data available for regime report")
            return None

        # 평균 regime score 계산
        scores: list[float] = []
        for sym in symbol_snapshots:
            score = compute_regime_score(
                funding_rate=sym.funding_rate,
                oi_change_pct=0.0,  # 단일 스냅샷이므로 변화율 계산 불가
                ls_ratio=sym.ls_ratio,
                taker_ratio=sym.taker_ratio,
            )
            scores.append(score)

        avg_score = sum(scores) / len(scores)
        label = classify_regime(avg_score)

        return MarketRegimeReport(
            timestamp=datetime.now(UTC),
            regime_score=avg_score,
            regime_label=label,
            symbols=tuple(symbol_snapshots),
        )

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

    # ─── Private Helpers ──────────────────────────────────────

    def _count_onchain_sources(self) -> tuple[int, int]:
        """Prometheus gauge에서 on-chain source staleness 판정.

        Returns:
            (total, ok) 튜플. 48시간 이내 fetch = OK.
        """
        stale_threshold = 48 * 3600  # 48시간
        try:
            from src.monitoring.metrics import onchain_last_success_gauge

            metrics = list(onchain_last_success_gauge.collect())
            if not metrics or not metrics[0].samples:
                return 0, 0

            now = time.time()
            total = len(metrics[0].samples)
            ok = sum(1 for s in metrics[0].samples if (now - s.value) < stale_threshold)
        except ImportError:
            return 0, 0
        else:
            return total, ok
