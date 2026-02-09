"""Vol-Adaptive Trend Strategy.

EMA crossover + RSI confirm + ADX filter + ATR vol-target sizing.
"""

from src.strategy.vol_adaptive.config import ShortMode, VolAdaptiveConfig
from src.strategy.vol_adaptive.preprocessor import preprocess
from src.strategy.vol_adaptive.signal import generate_signals
from src.strategy.vol_adaptive.strategy import VolAdaptiveStrategy

__all__ = [
    "ShortMode",
    "VolAdaptiveConfig",
    "VolAdaptiveStrategy",
    "generate_signals",
    "preprocess",
]
