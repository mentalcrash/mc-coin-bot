"""Database 연결 관리자 — aiosqlite WAL 모드.

단일 aiosqlite.Connection을 관리하며, 스키마 자동 생성을 포함합니다.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
from loguru import logger

from src.eda.persistence.schema import SCHEMA_SQL


class Database:
    """aiosqlite 연결 수명 관리자.

    Args:
        db_path: SQLite 파일 경로. ":memory:" 시 인메모리 DB (테스트용).
    """

    def __init__(self, db_path: str = "data/trading.db") -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """DB 연결 + WAL 모드 + 스키마 생성."""
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = await aiosqlite.connect(self._db_path)
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._create_schema()
        logger.info("Database connected: {}", self._db_path)

    async def close(self) -> None:
        """DB 연결 종료."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            logger.info("Database closed: {}", self._db_path)

    @property
    def connection(self) -> aiosqlite.Connection:
        """활성 연결 반환. 연결 안 됐으면 AssertionError."""
        assert self._conn is not None, "Database not connected. Call connect() first."
        return self._conn

    async def _create_schema(self) -> None:
        """멱등 스키마 생성."""
        assert self._conn is not None
        await self._conn.executescript(SCHEMA_SQL)
        await self._conn.commit()
