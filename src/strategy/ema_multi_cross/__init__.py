"""EMA Multi-Cross — 3쌍 EMA 크로스 합의 투표 전략."""

from src.strategy.ema_multi_cross.config import EmaMultiCrossConfig, ShortMode
from src.strategy.ema_multi_cross.preprocessor import preprocess
from src.strategy.ema_multi_cross.signal import generate_signals
from src.strategy.ema_multi_cross.strategy import EmaMultiCrossStrategy

__all__ = [
    "EmaMultiCrossConfig",
    "EmaMultiCrossStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
