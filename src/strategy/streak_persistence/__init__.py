"""Return Streak Persistence: 연속 양봉/음봉 군집 행동 기반 추세 지속 포착."""

from src.strategy.streak_persistence.config import ShortMode, StreakPersistenceConfig
from src.strategy.streak_persistence.preprocessor import preprocess
from src.strategy.streak_persistence.signal import generate_signals
from src.strategy.streak_persistence.strategy import StreakPersistenceStrategy

__all__ = [
    "ShortMode",
    "StreakPersistenceConfig",
    "StreakPersistenceStrategy",
    "generate_signals",
    "preprocess",
]
