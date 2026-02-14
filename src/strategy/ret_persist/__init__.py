"""Return Persistence Score: positive return bar ratio for trend persistence."""

from src.strategy.ret_persist.config import RetPersistConfig, ShortMode
from src.strategy.ret_persist.preprocessor import preprocess
from src.strategy.ret_persist.signal import generate_signals
from src.strategy.ret_persist.strategy import RetPersistStrategy

__all__ = [
    "RetPersistConfig",
    "RetPersistStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
