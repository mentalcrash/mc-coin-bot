"""TradePersistence 테스트 — 이벤트 → SQLite 영속화."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from src.core.event_bus import EventBus
from src.core.events import (
    BalanceUpdateEvent,
    CircuitBreakerEvent,
    FillEvent,
    OrderRejectedEvent,
    OrderRequestEvent,
    PositionUpdateEvent,
    RiskAlertEvent,
)
from src.eda.persistence.database import Database
from src.eda.persistence.trade_persistence import TradePersistence
from src.models.types import Direction


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    """인메모리 DB fixture."""
    database = Database(":memory:")
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def persistence(db: Database) -> TradePersistence:
    """TradePersistence fixture."""
    return TradePersistence(db, strategy_name="test-strategy")


@pytest.fixture
async def bus() -> EventBus:
    """EventBus fixture."""
    return EventBus(queue_size=1000)


def _ts() -> datetime:
    return datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


class TestFillEvent:
    """FillEvent → trades 테이블 테스트."""

    @pytest.mark.asyncio
    async def test_fill_creates_trade_row(
        self, db: Database, persistence: TradePersistence, bus: EventBus
    ) -> None:
        """FillEvent가 trades 테이블에 FILLED 행을 생성."""
        await persistence.register(bus)
        bus_task = asyncio.create_task(bus.start())

        fill = FillEvent(
            client_order_id="order-1",
            symbol="BTC/USDT",
            side="BUY",
            fill_price=50000.0,
            fill_qty=0.1,
            fee=5.0,
            fill_timestamp=_ts(),
        )
        await bus.publish(fill)
        await bus.flush()
        # fire-and-forget task 완료 대기
        await asyncio.sleep(0.05)

        trades = await persistence.get_recent_trades()
        assert len(trades) == 1
        assert trades[0]["client_order_id"] == "order-1"
        assert trades[0]["status"] == "FILLED"
        assert trades[0]["fill_price"] == 50000.0
        assert trades[0]["strategy_name"] == "test-strategy"

        await bus.stop()
        await bus_task

    @pytest.mark.asyncio
    async def test_order_request_enriches_fill(
        self, db: Database, persistence: TradePersistence, bus: EventBus
    ) -> None:
        """OrderRequest(validated=True) → Fill 보강 (target_weight 존재)."""
        await persistence.register(bus)
        bus_task = asyncio.create_task(bus.start())

        order = OrderRequestEvent(
            client_order_id="order-2",
            symbol="ETH/USDT",
            side="BUY",
            target_weight=0.5,
            notional_usd=5000.0,
            validated=True,
        )
        await bus.publish(order)
        await bus.flush()

        fill = FillEvent(
            client_order_id="order-2",
            symbol="ETH/USDT",
            side="BUY",
            fill_price=3000.0,
            fill_qty=1.0,
            fee=3.0,
            fill_timestamp=_ts(),
        )
        await bus.publish(fill)
        await bus.flush()
        await asyncio.sleep(0.05)

        trades = await persistence.get_recent_trades()
        assert len(trades) == 1
        assert trades[0]["target_weight"] == 0.5
        assert trades[0]["notional_usd"] == 5000.0

        await bus.stop()
        await bus_task


class TestOrderRejected:
    """OrderRejectedEvent → trades REJECTED 테스트."""

    @pytest.mark.asyncio
    async def test_rejected_order_creates_trade_row(
        self, db: Database, persistence: TradePersistence, bus: EventBus
    ) -> None:
        """OrderRejectedEvent가 REJECTED 상태 행을 생성."""
        await persistence.register(bus)
        bus_task = asyncio.create_task(bus.start())

        rejected = OrderRejectedEvent(
            client_order_id="order-rej-1",
            symbol="BTC/USDT",
            reason="Leverage exceeded",
        )
        await bus.publish(rejected)
        await bus.flush()
        await asyncio.sleep(0.05)

        trades = await persistence.get_recent_trades()
        assert len(trades) == 1
        assert trades[0]["status"] == "REJECTED"

        await bus.stop()
        await bus_task


class TestBalanceUpdate:
    """BalanceUpdateEvent → equity_snapshots 테스트."""

    @pytest.mark.asyncio
    async def test_balance_creates_equity_row(
        self, db: Database, persistence: TradePersistence, bus: EventBus
    ) -> None:
        """BalanceUpdateEvent가 equity_snapshots 행을 생성."""
        await persistence.register(bus)
        bus_task = asyncio.create_task(bus.start())

        balance = BalanceUpdateEvent(
            total_equity=10500.0,
            available_cash=8000.0,
            total_margin_used=2500.0,
        )
        await bus.publish(balance)
        await bus.flush()
        await asyncio.sleep(0.05)

        curve = await persistence.get_equity_curve()
        assert len(curve) == 1
        assert curve[0]["total_equity"] == 10500.0
        assert curve[0]["available_cash"] == 8000.0

        await bus.stop()
        await bus_task


class TestPositionUpdate:
    """PositionUpdateEvent → positions_history 테스트."""

    @pytest.mark.asyncio
    async def test_position_creates_history_row(
        self, db: Database, persistence: TradePersistence, bus: EventBus
    ) -> None:
        """PositionUpdateEvent가 positions_history 행을 생성."""
        await persistence.register(bus)
        bus_task = asyncio.create_task(bus.start())

        pos = PositionUpdateEvent(
            symbol="SOL/USDT",
            direction=Direction.LONG,
            size=10.0,
            avg_entry_price=150.0,
            unrealized_pnl=50.0,
            realized_pnl=0.0,
        )
        await bus.publish(pos)
        await bus.flush()
        await asyncio.sleep(0.05)

        conn = db.connection
        cursor = await conn.execute("SELECT * FROM positions_history")
        rows = await cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][1] == "SOL/USDT"  # symbol
        assert rows[0][2] == Direction.LONG.value  # direction

        await bus.stop()
        await bus_task


class TestRiskEvents:
    """CircuitBreaker / RiskAlert → risk_events 테스트."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_creates_risk_row(
        self, db: Database, persistence: TradePersistence, bus: EventBus
    ) -> None:
        """CircuitBreakerEvent가 risk_events 행을 생성."""
        await persistence.register(bus)
        bus_task = asyncio.create_task(bus.start())

        cb = CircuitBreakerEvent(
            reason="Drawdown 15% exceeded",
            close_all_positions=True,
        )
        await bus.publish(cb)
        await bus.flush()
        await asyncio.sleep(0.05)

        conn = db.connection
        cursor = await conn.execute("SELECT * FROM risk_events")
        rows = await cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][1] == "CIRCUIT_BREAKER"  # event_type

        await bus.stop()
        await bus_task

    @pytest.mark.asyncio
    async def test_risk_alert_creates_risk_row(
        self, db: Database, persistence: TradePersistence, bus: EventBus
    ) -> None:
        """RiskAlertEvent가 risk_events 행을 생성."""
        await persistence.register(bus)
        bus_task = asyncio.create_task(bus.start())

        alert = RiskAlertEvent(
            alert_level="WARNING",
            message="Leverage approaching limit",
        )
        await bus.publish(alert)
        await bus.flush()
        await asyncio.sleep(0.05)

        conn = db.connection
        cursor = await conn.execute("SELECT * FROM risk_events")
        rows = await cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][1] == "RISK_ALERT"

        await bus.stop()
        await bus_task


class TestNonBlocking:
    """핸들러 non-blocking 동작 테스트."""

    @pytest.mark.asyncio
    async def test_handler_returns_immediately(
        self, db: Database, persistence: TradePersistence, bus: EventBus
    ) -> None:
        """핸들러가 즉시 반환하는지 확인 (DB 쓰기 대기 안함)."""
        await persistence.register(bus)
        bus_task = asyncio.create_task(bus.start())

        fill = FillEvent(
            client_order_id="fast-1",
            symbol="BTC/USDT",
            side="BUY",
            fill_price=50000.0,
            fill_qty=0.1,
            fee=0.0,
            fill_timestamp=_ts(),
        )
        await bus.publish(fill)
        # flush만으로도 handler는 즉시 리턴해야 함
        await bus.flush()

        await bus.stop()
        await bus_task


class TestDbErrorIsolation:
    """DB 오류 시 예외 미전파 테스트."""

    @pytest.mark.asyncio
    async def test_db_error_does_not_propagate(self, bus: EventBus) -> None:
        """DB 연결 끊긴 상태에서도 핸들러가 예외를 전파하지 않는지 확인."""
        database = Database(":memory:")
        await database.connect()
        persistence = TradePersistence(database, strategy_name="test")
        await persistence.register(bus)
        bus_task = asyncio.create_task(bus.start())

        # DB 강제 닫기
        await database.close()

        fill = FillEvent(
            client_order_id="err-1",
            symbol="BTC/USDT",
            side="BUY",
            fill_price=50000.0,
            fill_qty=0.1,
            fee=0.0,
            fill_timestamp=_ts(),
        )
        await bus.publish(fill)
        await bus.flush()
        await asyncio.sleep(0.05)

        # 예외 없이 통과
        await bus.stop()
        await bus_task


class TestQueryMethods:
    """조회 메서드 테스트."""

    @pytest.mark.asyncio
    async def test_get_recent_trades_with_symbol_filter(
        self, db: Database, persistence: TradePersistence, bus: EventBus
    ) -> None:
        """심볼 필터가 적용된 get_recent_trades."""
        await persistence.register(bus)
        bus_task = asyncio.create_task(bus.start())

        for sym in ["BTC/USDT", "ETH/USDT"]:
            fill = FillEvent(
                client_order_id=f"order-{sym}",
                symbol=sym,
                side="BUY",
                fill_price=50000.0,
                fill_qty=0.1,
                fee=0.0,
                fill_timestamp=_ts(),
            )
            await bus.publish(fill)

        await bus.flush()
        await asyncio.sleep(0.05)

        btc_trades = await persistence.get_recent_trades(symbol="BTC/USDT")
        assert len(btc_trades) == 1
        assert btc_trades[0]["symbol"] == "BTC/USDT"

        all_trades = await persistence.get_recent_trades()
        assert len(all_trades) == 2

        await bus.stop()
        await bus_task

    @pytest.mark.asyncio
    async def test_get_equity_curve_with_limit(
        self, db: Database, persistence: TradePersistence, bus: EventBus
    ) -> None:
        """get_equity_curve limit 파라미터 동작 확인."""
        await persistence.register(bus)
        bus_task = asyncio.create_task(bus.start())

        for i in range(5):
            balance = BalanceUpdateEvent(
                total_equity=10000.0 + i * 100,
                available_cash=10000.0 + i * 100,
            )
            await bus.publish(balance)
            await bus.flush()

        await asyncio.sleep(0.05)

        full_curve = await persistence.get_equity_curve(limit=1000)
        assert len(full_curve) == 5
        limited = await persistence.get_equity_curve(limit=2)
        assert len(limited) == 2

        await bus.stop()
        await bus_task


class TestUnvalidatedOrderNotCached:
    """validated=False OrderRequest는 캐시하지 않는지 테스트."""

    @pytest.mark.asyncio
    async def test_unvalidated_order_not_cached(
        self, db: Database, persistence: TradePersistence, bus: EventBus
    ) -> None:
        """validated=False인 OrderRequest는 _pending_orders에 캐시되지 않아야 함."""
        await persistence.register(bus)
        bus_task = asyncio.create_task(bus.start())

        order = OrderRequestEvent(
            client_order_id="unval-1",
            symbol="BTC/USDT",
            side="BUY",
            target_weight=0.5,
            notional_usd=5000.0,
            validated=False,
        )
        await bus.publish(order)
        await bus.flush()

        # validated=False이므로 캐시에 없어야 함
        assert "unval-1" not in persistence._pending_orders

        # fill 발행 시 target_weight 보강 없이 None
        fill = FillEvent(
            client_order_id="unval-1",
            symbol="BTC/USDT",
            side="BUY",
            fill_price=50000.0,
            fill_qty=0.1,
            fee=0.0,
            fill_timestamp=_ts(),
        )
        await bus.publish(fill)
        await bus.flush()
        await asyncio.sleep(0.05)

        trades = await persistence.get_recent_trades()
        assert len(trades) == 1
        assert trades[0]["target_weight"] is None

        await bus.stop()
        await bus_task


class TestIdempotentInsert:
    """INSERT OR REPLACE 멱등성 테스트."""

    @pytest.mark.asyncio
    async def test_duplicate_fill_overwrites(
        self, db: Database, persistence: TradePersistence, bus: EventBus
    ) -> None:
        """동일 client_order_id로 FillEvent 2회 발행 시 행 1개만 존재."""
        await persistence.register(bus)
        bus_task = asyncio.create_task(bus.start())

        for price in [50000.0, 51000.0]:
            fill = FillEvent(
                client_order_id="dup-1",
                symbol="BTC/USDT",
                side="BUY",
                fill_price=price,
                fill_qty=0.1,
                fee=0.0,
                fill_timestamp=_ts(),
            )
            await bus.publish(fill)
            await bus.flush()
            await asyncio.sleep(0.05)

        trades = await persistence.get_recent_trades()
        assert len(trades) == 1
        # 마지막 INSERT OR REPLACE가 적용됨
        assert trades[0]["fill_price"] == 51000.0

        await bus.stop()
        await bus_task


class TestFillFieldCompleteness:
    """FillEvent 전체 필드 저장 확인."""

    @pytest.mark.asyncio
    async def test_all_fill_fields_stored(
        self, db: Database, persistence: TradePersistence, bus: EventBus
    ) -> None:
        """FillEvent의 모든 필드가 trades 행에 정확히 저장되는지 확인."""
        await persistence.register(bus)
        bus_task = asyncio.create_task(bus.start())

        ts = datetime(2025, 6, 15, 14, 30, 0, tzinfo=UTC)
        fill = FillEvent(
            client_order_id="full-1",
            symbol="SOL/USDT",
            side="SELL",
            fill_price=155.5,
            fill_qty=20.0,
            fee=3.11,
            fill_timestamp=ts,
        )
        await bus.publish(fill)
        await bus.flush()
        await asyncio.sleep(0.05)

        trades = await persistence.get_recent_trades()
        assert len(trades) == 1
        t = trades[0]
        assert t["client_order_id"] == "full-1"
        assert t["symbol"] == "SOL/USDT"
        assert t["side"] == "SELL"
        assert t["fill_price"] == 155.5
        assert t["fill_qty"] == 20.0
        assert t["fee"] == 3.11
        assert t["status"] == "FILLED"
        assert t["strategy_name"] == "test-strategy"
        assert t["timestamp"] == ts.isoformat()

        await bus.stop()
        await bus_task


class TestMultipleEventsBurst:
    """다수 이벤트 연속 발행 시 데이터 무결성."""

    @pytest.mark.asyncio
    async def test_burst_events_all_saved(
        self, db: Database, persistence: TradePersistence, bus: EventBus
    ) -> None:
        """10개 FillEvent를 연속 발행하면 모두 저장되는지 확인."""
        await persistence.register(bus)
        bus_task = asyncio.create_task(bus.start())

        n = 10
        for i in range(n):
            fill = FillEvent(
                client_order_id=f"burst-{i}",
                symbol="BTC/USDT",
                side="BUY",
                fill_price=50000.0 + i,
                fill_qty=0.01,
                fee=0.0,
                fill_timestamp=_ts(),
            )
            await bus.publish(fill)

        await bus.flush()
        await asyncio.sleep(0.1)

        trades = await persistence.get_recent_trades(limit=100)
        assert len(trades) == n

        await bus.stop()
        await bus_task
