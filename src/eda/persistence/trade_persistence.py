"""TradePersistence — EventBus subscriber → SQLite 영속화.

거래 이벤트를 구독하여 SQLite에 비동기로 기록합니다.
DB 쓰기는 asyncio.create_task()로 fire-and-forget하여 이벤트 체인을 차단하지 않습니다.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from src.core.events import (
    AnyEvent,
    BalanceUpdateEvent,
    CircuitBreakerEvent,
    EventType,
    FillEvent,
    OrderRejectedEvent,
    OrderRequestEvent,
    PositionUpdateEvent,
    RiskAlertEvent,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from src.core.event_bus import EventBus
    from src.eda.persistence.database import Database


class TradePersistence:
    """EventBus subscriber — 거래 데이터를 SQLite에 영속화.

    Args:
        database: Database 인스턴스 (연결 완료 상태)
        strategy_name: 전략 이름 (trades 테이블에 기록)
    """

    def __init__(self, database: Database, strategy_name: str = "unknown") -> None:
        self._db = database
        self._strategy_name = strategy_name
        self._pending_orders: dict[str, OrderRequestEvent] = {}
        self._background_tasks: set[asyncio.Task[None]] = set()

    async def register(self, bus: EventBus) -> None:
        """EventBus에 7개 이벤트 핸들러 등록."""
        bus.subscribe(EventType.FILL, self._on_fill)
        bus.subscribe(EventType.ORDER_REQUEST, self._on_order_request)
        bus.subscribe(EventType.ORDER_REJECTED, self._on_order_rejected)
        bus.subscribe(EventType.BALANCE_UPDATE, self._on_balance_update)
        bus.subscribe(EventType.POSITION_UPDATE, self._on_position_update)
        bus.subscribe(EventType.CIRCUIT_BREAKER, self._on_circuit_breaker)
        bus.subscribe(EventType.RISK_ALERT, self._on_risk_alert)

    # =========================================================================
    # Event Handlers (fire-and-forget)
    # =========================================================================

    async def _on_order_request(self, event: AnyEvent) -> None:
        """validated=True인 주문을 캐시 (fill 보강용)."""
        assert isinstance(event, OrderRequestEvent)
        if event.validated:
            self._pending_orders[event.client_order_id] = event

    async def _on_fill(self, event: AnyEvent) -> None:
        """FillEvent → trades INSERT OR REPLACE."""
        assert isinstance(event, FillEvent)
        self._spawn(self._save_trade(event))

    async def _on_order_rejected(self, event: AnyEvent) -> None:
        """OrderRejectedEvent → trades INSERT (REJECTED)."""
        assert isinstance(event, OrderRejectedEvent)
        self._spawn(self._save_rejected_order(event))

    async def _on_balance_update(self, event: AnyEvent) -> None:
        """BalanceUpdateEvent → equity_snapshots INSERT."""
        assert isinstance(event, BalanceUpdateEvent)
        self._spawn(self._save_equity_snapshot(event))

    async def _on_position_update(self, event: AnyEvent) -> None:
        """PositionUpdateEvent → positions_history INSERT."""
        assert isinstance(event, PositionUpdateEvent)
        self._spawn(self._save_position(event))

    async def _on_circuit_breaker(self, event: AnyEvent) -> None:
        """CircuitBreakerEvent → risk_events INSERT."""
        assert isinstance(event, CircuitBreakerEvent)
        self._spawn(self._save_risk_event_cb(event))

    async def _on_risk_alert(self, event: AnyEvent) -> None:
        """RiskAlertEvent → risk_events INSERT."""
        assert isinstance(event, RiskAlertEvent)
        self._spawn(self._save_risk_event_alert(event))

    def _spawn(self, coro: Coroutine[object, object, None]) -> None:
        """fire-and-forget task 생성 (GC 방지용 참조 보관)."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    # =========================================================================
    # DB Write Methods (try/except 격리)
    # =========================================================================

    async def _save_trade(self, fill: FillEvent) -> None:
        """FillEvent → trades 테이블 INSERT OR REPLACE."""
        try:
            # pending order에서 보강 정보 가져오기
            pending = self._pending_orders.pop(fill.client_order_id, None)
            target_weight = pending.target_weight if pending else None
            notional_usd = pending.notional_usd if pending else None
            corr_id = str(fill.correlation_id) if fill.correlation_id else None

            conn = self._db.connection
            await conn.execute(
                """INSERT OR REPLACE INTO trades
                   (client_order_id, symbol, side, fill_price, fill_qty, fee,
                    target_weight, notional_usd, status, strategy_name, timestamp, correlation_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'FILLED', ?, ?, ?)""",
                (
                    fill.client_order_id,
                    fill.symbol,
                    fill.side,
                    fill.fill_price,
                    fill.fill_qty,
                    fill.fee,
                    target_weight,
                    notional_usd,
                    self._strategy_name,
                    fill.fill_timestamp.isoformat(),
                    corr_id,
                ),
            )
            await conn.commit()
        except Exception:
            logger.exception("Failed to save trade: {}", fill.client_order_id)

    async def _save_rejected_order(self, event: OrderRejectedEvent) -> None:
        """OrderRejectedEvent → trades 테이블 INSERT (REJECTED)."""
        try:
            conn = self._db.connection
            corr_id = str(event.correlation_id) if event.correlation_id else None
            await conn.execute(
                """INSERT OR REPLACE INTO trades
                   (client_order_id, symbol, side, status, strategy_name, timestamp, correlation_id)
                   VALUES (?, ?, '', 'REJECTED', ?, ?, ?)""",
                (
                    event.client_order_id,
                    event.symbol,
                    self._strategy_name,
                    event.timestamp.isoformat(),
                    corr_id,
                ),
            )
            await conn.commit()
        except Exception:
            logger.exception("Failed to save rejected order: {}", event.client_order_id)

    async def _save_equity_snapshot(self, event: BalanceUpdateEvent) -> None:
        """BalanceUpdateEvent → equity_snapshots 테이블 INSERT."""
        try:
            conn = self._db.connection
            await conn.execute(
                """INSERT INTO equity_snapshots
                   (total_equity, available_cash, margin_used, timestamp)
                   VALUES (?, ?, ?, ?)""",
                (
                    event.total_equity,
                    event.available_cash,
                    event.total_margin_used,
                    event.timestamp.isoformat(),
                ),
            )
            await conn.commit()
        except Exception:
            logger.exception("Failed to save equity snapshot")

    async def _save_position(self, event: PositionUpdateEvent) -> None:
        """PositionUpdateEvent → positions_history 테이블 INSERT."""
        try:
            conn = self._db.connection
            await conn.execute(
                """INSERT INTO positions_history
                   (symbol, direction, size, avg_entry_price, unrealized_pnl, realized_pnl, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.symbol,
                    event.direction.value,
                    event.size,
                    event.avg_entry_price,
                    event.unrealized_pnl,
                    event.realized_pnl,
                    event.timestamp.isoformat(),
                ),
            )
            await conn.commit()
        except Exception:
            logger.exception("Failed to save position: {}", event.symbol)

    async def _save_risk_event_cb(self, event: CircuitBreakerEvent) -> None:
        """CircuitBreakerEvent → risk_events 테이블 INSERT."""
        try:
            conn = self._db.connection
            await conn.execute(
                """INSERT INTO risk_events
                   (event_type, reason, close_all, timestamp)
                   VALUES ('CIRCUIT_BREAKER', ?, ?, ?)""",
                (
                    event.reason,
                    1 if event.close_all_positions else 0,
                    event.timestamp.isoformat(),
                ),
            )
            await conn.commit()
        except Exception:
            logger.exception("Failed to save circuit breaker event")

    async def _save_risk_event_alert(self, event: RiskAlertEvent) -> None:
        """RiskAlertEvent → risk_events 테이블 INSERT."""
        try:
            conn = self._db.connection
            await conn.execute(
                """INSERT INTO risk_events
                   (event_type, message, alert_level, timestamp)
                   VALUES ('RISK_ALERT', ?, ?, ?)""",
                (
                    event.message,
                    event.alert_level,
                    event.timestamp.isoformat(),
                ),
            )
            await conn.commit()
        except Exception:
            logger.exception("Failed to save risk alert")

    # =========================================================================
    # Query Methods
    # =========================================================================

    async def get_recent_trades(
        self, symbol: str | None = None, limit: int = 50
    ) -> list[dict[str, object]]:
        """최근 거래 조회.

        Args:
            symbol: 필터 심볼 (None이면 전체)
            limit: 최대 행 수

        Returns:
            거래 행 목록 (dict)
        """
        conn = self._db.connection
        if symbol:
            cursor = await conn.execute(
                "SELECT * FROM trades WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                (symbol, limit),
            )
        else:
            cursor = await conn.execute(
                "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
        columns = [desc[0] for desc in cursor.description or []]
        rows = await cursor.fetchall()
        return [dict(zip(columns, row, strict=True)) for row in rows]

    async def get_equity_curve(self, limit: int = 1000) -> list[dict[str, object]]:
        """equity 스냅샷 조회.

        Args:
            limit: 최대 행 수

        Returns:
            equity 스냅샷 목록 (dict)
        """
        conn = self._db.connection
        cursor = await conn.execute(
            "SELECT * FROM equity_snapshots ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        columns = [desc[0] for desc in cursor.description or []]
        rows = await cursor.fetchall()
        return [dict(zip(columns, row, strict=True)) for row in rows]
