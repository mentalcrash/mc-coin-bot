"""데이터 영속화 패키지 — SQLite via aiosqlite.

TradePersistence(이벤트 기록)와 StateManager(상태 저장/복구)를 제공합니다.
"""

from src.eda.persistence.database import Database
from src.eda.persistence.state_manager import StateManager
from src.eda.persistence.trade_persistence import TradePersistence

__all__ = ["Database", "StateManager", "TradePersistence"]
