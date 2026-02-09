"""StateManager — 봇 상태 저장/복구.

bot_state 테이블에 PM/RM 상태를 JSON으로 직렬화하여 저장합니다.
재시작 시 상태를 복구하여 포지션과 리스크 상태를 유지합니다.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.eda.persistence.database import Database
    from src.eda.portfolio_manager import EDAPortfolioManager
    from src.eda.risk_manager import EDARiskManager

# bot_state 테이블 키 상수
_KEY_PM_STATE = "pm_state"
_KEY_RM_STATE = "rm_state"
_KEY_LAST_SAVE = "last_save_timestamp"


class StateManager:
    """봇 상태 저장/복구 — bot_state 테이블 활용.

    Args:
        database: Database 인스턴스 (연결 완료 상태)
    """

    def __init__(self, database: Database) -> None:
        self._db = database

    # =========================================================================
    # 저장
    # =========================================================================

    async def save_pm_state(self, pm: EDAPortfolioManager) -> None:
        """PM 상태를 bot_state에 저장."""
        positions_data: dict[str, dict[str, object]] = {}
        for symbol, pos in pm.positions.items():
            positions_data[symbol] = {
                "direction": pos.direction.value,
                "size": pos.size,
                "avg_entry_price": pos.avg_entry_price,
                "realized_pnl": pos.realized_pnl,
                "unrealized_pnl": pos.unrealized_pnl,
                "current_weight": pos.current_weight,
                "last_price": pos.last_price,
                "peak_price_since_entry": pos.peak_price_since_entry,
                "trough_price_since_entry": pos.trough_price_since_entry,
                "atr_values": pos.atr_values,
            }

        state = {
            "positions": positions_data,
            "cash": pm.available_cash,
            "order_counter": pm.order_counter,
            "last_target_weights": pm.last_target_weights,
            "last_executed_targets": pm.last_executed_targets,
        }
        await self._save_key(_KEY_PM_STATE, json.dumps(state))

    async def save_rm_state(self, rm: EDARiskManager) -> None:
        """RM 상태를 bot_state에 저장."""
        state = {
            "peak_equity": rm.peak_equity,
            "circuit_breaker_triggered": rm.is_circuit_breaker_active,
        }
        await self._save_key(_KEY_RM_STATE, json.dumps(state))

    async def save_all(self, pm: EDAPortfolioManager, rm: EDARiskManager) -> None:
        """PM + RM 상태를 한 번에 저장."""
        await self.save_pm_state(pm)
        await self.save_rm_state(rm)
        await self._save_key(_KEY_LAST_SAVE, datetime.now(UTC).isoformat())
        logger.debug("State saved (PM + RM)")

    # =========================================================================
    # 로드
    # =========================================================================

    async def load_pm_state(self) -> dict[str, object] | None:
        """저장된 PM 상태를 로드. 없으면 None."""
        raw = await self._load_key(_KEY_PM_STATE)
        if raw is None:
            return None
        return json.loads(raw)  # type: ignore[no-any-return]

    async def load_rm_state(self) -> dict[str, object] | None:
        """저장된 RM 상태를 로드. 없으면 None."""
        raw = await self._load_key(_KEY_RM_STATE)
        if raw is None:
            return None
        return json.loads(raw)  # type: ignore[no-any-return]

    async def get_last_save_timestamp(self) -> datetime | None:
        """마지막 저장 시각. 없으면 None."""
        raw = await self._load_key(_KEY_LAST_SAVE)
        if raw is None:
            return None
        return datetime.fromisoformat(raw)

    # =========================================================================
    # 유틸리티
    # =========================================================================

    async def clear_state(self) -> None:
        """모든 상태 삭제."""
        conn = self._db.connection
        await conn.execute("DELETE FROM bot_state")
        await conn.commit()
        logger.info("Bot state cleared")

    # =========================================================================
    # Internal helpers
    # =========================================================================

    async def _save_key(self, key: str, value: str) -> None:
        """bot_state에 key-value INSERT OR REPLACE."""
        conn = self._db.connection
        now = datetime.now(UTC).isoformat()
        await conn.execute(
            "INSERT OR REPLACE INTO bot_state (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, now),
        )
        await conn.commit()

    async def _load_key(self, key: str) -> str | None:
        """bot_state에서 key로 value 조회."""
        conn = self._db.connection
        cursor = await conn.execute(
            "SELECT value FROM bot_state WHERE key = ?",
            (key,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return row[0]  # type: ignore[no-any-return]
