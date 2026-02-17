"""HealthCheckScheduler — 주기적 시스템/마켓/전략 건강 상태 알림.

3개 내부 asyncio 루프:
- heartbeat_loop (1시간 주기): 시스템 생존 확인
- regime_loop (4시간 주기): 마켓 regime 리포트
- strategy_health_loop (8시간 주기): 전략 건강도 확인 + alpha decay 감지

ReportScheduler 패턴을 미러링합니다.

Rules Applied:
    - EDA 패턴: asyncio task lifecycle
    - #10 Python Standards: Async patterns, type hints
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.data.derivatives_snapshot import DerivativesSnapshotFetcher
from src.data.regime_score import classify_regime, compute_regime_score
from src.notification.health_formatters import (
    format_heartbeat_embed,
    format_regime_embed,
    format_strategy_health_embed,
)
from src.notification.health_models import (
    MarketRegimeReport,
    PositionStatus,
    StrategyHealthSnapshot,
    StrategyPerformanceSnapshot,
    SystemHealthSnapshot,
)
from src.notification.models import ChannelRoute, NotificationItem, Severity

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.event_bus import EventBus
    from src.eda.analytics import AnalyticsEngine
    from src.eda.live_data_feed import LiveDataFeed
    from src.eda.portfolio_manager import EDAPortfolioManager
    from src.eda.risk_manager import EDARiskManager
    from src.exchange.binance_futures_client import BinanceFuturesClient
    from src.notification.queue import NotificationQueue

# 루프 주기 (초)
_HEARTBEAT_INTERVAL = 3600.0  # 1시간
_REGIME_INTERVAL = 14400.0  # 4시간
_STRATEGY_HEALTH_INTERVAL = 28800.0  # 8시간

# Rolling Sharpe 계산 기간
_ROLLING_SHARPE_DAYS = 30
_RECENT_TRADES_COUNT = 20
_ALPHA_DECAY_WINDOW = 3
_ALPHA_DECAY_CONFIRMATIONS = 2
_MAX_SHARPE_HISTORY = 10
_SHARPE_HEALTHY_THRESHOLD = 0.5

# 연환산 계수 (일봉 기준)
_ANNUALIZE_SQRT = 365.0**0.5


class HealthCheckScheduler:
    """주기적 Health Check 스케줄러.

    Args:
        queue: NotificationQueue (enqueue 대상)
        pm: EDAPortfolioManager
        rm: EDARiskManager
        analytics: AnalyticsEngine
        feed: LiveDataFeed
        bus: EventBus
        futures_client: BinanceFuturesClient (None이면 내부 ccxt 생성)
        symbols: 모니터링 대상 심볼 리스트
    """

    def __init__(
        self,
        queue: NotificationQueue,
        pm: EDAPortfolioManager,
        rm: EDARiskManager,
        analytics: AnalyticsEngine,
        feed: LiveDataFeed,
        bus: EventBus,
        futures_client: BinanceFuturesClient | None,
        symbols: list[str],
        exchange_stop_mgr: Any = None,
    ) -> None:
        self._queue = queue
        self._pm = pm
        self._rm = rm
        self._analytics = analytics
        self._feed = feed
        self._bus = bus
        self._symbols = symbols
        self._exchange_stop_mgr = exchange_stop_mgr
        self._snapshot_fetcher = DerivativesSnapshotFetcher(futures_client)

        self._heartbeat_task: asyncio.Task[None] | None = None
        self._regime_task: asyncio.Task[None] | None = None
        self._strategy_health_task: asyncio.Task[None] | None = None

        self._start_time = time.monotonic()
        self._sharpe_history: list[float] = []
        self._alpha_decay_streak: int = 0

    async def start(self) -> None:
        """스케줄 task 시작."""
        await self._snapshot_fetcher.start()

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._regime_task = asyncio.create_task(self._regime_loop())
        self._strategy_health_task = asyncio.create_task(self._strategy_health_loop())

        logger.info("HealthCheckScheduler started (heartbeat/regime/strategy)")

    async def stop(self) -> None:
        """Task 취소 + snapshot fetcher 정리."""
        for task in (self._heartbeat_task, self._regime_task, self._strategy_health_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        await self._snapshot_fetcher.stop()
        logger.info("HealthCheckScheduler stopped")

    async def trigger_health_check(self) -> None:
        """즉시 heartbeat 전송 (Discord /health 명령용)."""
        try:
            await self._send_heartbeat()
            logger.info("Health check triggered manually")
        except Exception:
            logger.exception("Failed to trigger health check")

    # ─── Loops ────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """1시간 주기 System Heartbeat."""
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            try:
                await self._send_heartbeat()
            except Exception:
                logger.exception("Failed to send heartbeat")

    async def _regime_loop(self) -> None:
        """4시간 주기 Market Regime."""
        while True:
            await asyncio.sleep(_REGIME_INTERVAL)
            try:
                await self._send_regime_report()
            except Exception:
                logger.exception("Failed to send regime report")

    async def _strategy_health_loop(self) -> None:
        """8시간 주기 Strategy Health."""
        while True:
            await asyncio.sleep(_STRATEGY_HEALTH_INTERVAL)
            try:
                await self._send_strategy_health()
            except Exception:
                logger.exception("Failed to send strategy health")

    # ─── Data Collectors + Senders ────────────────────────────

    async def _send_heartbeat(self) -> None:
        """Tier 1: System Heartbeat 수집 + enqueue."""
        snapshot = self._collect_system_health()
        embed = format_heartbeat_embed(snapshot)

        item = NotificationItem(
            severity=Severity.INFO,
            channel=ChannelRoute.HEARTBEAT,
            embed=embed,
            spam_key="heartbeat",
        )
        await self._queue.enqueue(item)

    async def _send_regime_report(self) -> None:
        """Tier 2: Market Regime 수집 + enqueue."""
        symbol_snapshots = await self._snapshot_fetcher.fetch_all(self._symbols)

        if not symbol_snapshots:
            logger.warning("No derivatives data available for regime report")
            return

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

        report = MarketRegimeReport(
            timestamp=datetime.now(UTC),
            regime_score=avg_score,
            regime_label=label,
            symbols=tuple(symbol_snapshots),
        )
        embed = format_regime_embed(report)

        item = NotificationItem(
            severity=Severity.INFO,
            channel=ChannelRoute.MARKET_REGIME,
            embed=embed,
            spam_key="regime_report",
        )
        await self._queue.enqueue(item)

    async def _send_strategy_health(self) -> None:
        """Tier 3: Strategy Health 수집 + enqueue."""
        snapshot = self._collect_strategy_health()
        embed = format_strategy_health_embed(snapshot)

        item = NotificationItem(
            severity=Severity.WARNING if snapshot.alpha_decay_detected else Severity.INFO,
            channel=ChannelRoute.DAILY_REPORT,
            embed=embed,
            spam_key="strategy_health",
        )
        await self._queue.enqueue(item)

    # ─── Snapshot Collectors ──────────────────────────────────

    def _collect_system_health(self) -> SystemHealthSnapshot:
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

    def _collect_strategy_health(self) -> StrategyHealthSnapshot:
        """Analytics/PM/RM에서 전략 건강 상태 수집."""
        trades = self._analytics.closed_trades
        now = datetime.now(UTC)

        # Rolling Sharpe (30d)
        cutoff_30d = now - timedelta(days=_ROLLING_SHARPE_DAYS)
        recent_trades_30d = [t for t in trades if t.exit_time and t.exit_time >= cutoff_30d]
        rolling_sharpe = self._compute_rolling_sharpe(recent_trades_30d)

        # Alpha decay 감지
        self._sharpe_history.append(rolling_sharpe)
        if len(self._sharpe_history) > _MAX_SHARPE_HISTORY:
            self._sharpe_history = self._sharpe_history[-_MAX_SHARPE_HISTORY:]

        alpha_decay = self._detect_alpha_decay()

        # 최근 20건 win rate / profit factor
        recent_n = trades[-_RECENT_TRADES_COUNT:] if trades else []
        win_rate, profit_factor = self._compute_trade_stats(recent_n)

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
        strategy_breakdown = self._build_strategy_breakdown(recent_trades_30d)

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

    # ─── Helper Functions ─────────────────────────────────────

    def _build_strategy_breakdown(
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
            sharpe = self._compute_rolling_sharpe(strades)
            win_rate, _ = self._compute_trade_stats(strades)
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
    def _compute_rolling_sharpe(
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

    def _detect_alpha_decay(self) -> bool:
        """직전 3개 Sharpe가 연속 _ALPHA_DECAY_CONFIRMATIONS 회 하강 추세인지 감지.

        8h 주기 x 2회 확인 = 16h 지속해야 발동 (false positive 억제).
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
    def _compute_trade_stats(
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
