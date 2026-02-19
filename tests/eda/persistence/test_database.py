"""Database 테스트 — 스키마 생성, WAL 모드, 연결 수명."""

from __future__ import annotations

import pathlib
from collections.abc import AsyncIterator

import pytest

from src.eda.persistence.database import Database

# 앱 정의 테이블 목록 (sqlite_sequence 제외)
_APP_TABLES = {"bot_state", "equity_snapshots", "positions_history", "risk_events", "trades"}


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    """인메모리 DB fixture."""
    database = Database(":memory:")
    await database.connect()
    yield database
    await database.close()


class TestDatabase:
    """Database 연결 및 스키마 테스트."""

    @pytest.mark.asyncio
    async def test_creates_all_tables(self, db: Database) -> None:
        """5개 앱 테이블이 모두 생성되는지 확인."""
        conn = db.connection
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in await cursor.fetchall()}
        assert _APP_TABLES.issubset(tables)

    @pytest.mark.asyncio
    async def test_schema_is_idempotent(self, db: Database) -> None:
        """스키마 재생성이 오류 없이 멱등 동작하는지 확인."""
        await db._create_schema()
        conn = db.connection
        cursor = await conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cursor.fetchall()}
        assert _APP_TABLES.issubset(tables)

    @pytest.mark.asyncio
    async def test_wal_mode_on_file_db(self, tmp_path: pathlib.Path) -> None:
        """파일 기반 DB에서 WAL journal 모드가 활성화되는지 확인."""
        db_path = str(tmp_path / "test_wal.db")
        database = Database(db_path)
        await database.connect()
        conn = database.connection
        cursor = await conn.execute("PRAGMA journal_mode")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "wal"
        await database.close()

    @pytest.mark.asyncio
    async def test_close_clears_connection(self, db: Database) -> None:
        """close() 후 connection 접근 시 AssertionError."""
        await db.close()
        with pytest.raises(AssertionError, match="not connected"):
            _ = db.connection

    @pytest.mark.asyncio
    async def test_file_db_creates_directory(self, tmp_path: pathlib.Path) -> None:
        """파일 기반 DB가 부모 디렉토리를 자동 생성하는지 확인."""
        db_path = str(tmp_path / "sub" / "dir" / "test.db")
        database = Database(db_path)
        await database.connect()
        assert pathlib.Path(db_path).exists()
        await database.close()

    @pytest.mark.asyncio
    async def test_indexes_created(self, db: Database) -> None:
        """스키마 인덱스가 생성되는지 확인."""
        conn = db.connection
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        )
        indexes = {row[0] for row in await cursor.fetchall()}
        expected = {
            "idx_trades_symbol",
            "idx_trades_timestamp",
            "idx_equity_timestamp",
            "idx_positions_symbol",
        }
        assert expected.issubset(indexes)

    @pytest.mark.asyncio
    async def test_synchronous_normal(self, tmp_path: pathlib.Path) -> None:
        """PRAGMA synchronous=NORMAL 설정 확인 (파일 DB)."""
        db_path = str(tmp_path / "sync_test.db")
        database = Database(db_path)
        await database.connect()
        conn = database.connection
        cursor = await conn.execute("PRAGMA synchronous")
        row = await cursor.fetchone()
        assert row is not None
        # NORMAL = 1
        assert row[0] == 1
        await database.close()

    @pytest.mark.asyncio
    async def test_trades_table_columns(self, db: Database) -> None:
        """trades 테이블 컬럼 구조 확인."""
        conn = db.connection
        cursor = await conn.execute("PRAGMA table_info(trades)")
        columns = {row[1] for row in await cursor.fetchall()}
        expected = {
            "client_order_id",
            "symbol",
            "side",
            "fill_price",
            "fill_qty",
            "fee",
            "target_weight",
            "notional_usd",
            "status",
            "strategy_name",
            "timestamp",
            "correlation_id",
        }
        assert expected == columns

    @pytest.mark.asyncio
    async def test_bot_state_table_columns(self, db: Database) -> None:
        """bot_state 테이블 컬럼 구조 확인."""
        conn = db.connection
        cursor = await conn.execute("PRAGMA table_info(bot_state)")
        columns = {row[1] for row in await cursor.fetchall()}
        assert columns == {"key", "value", "updated_at"}
