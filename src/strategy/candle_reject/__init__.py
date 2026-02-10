"""Candlestick Rejection Momentum Strategy.

4H candle rejection wick -> directional momentum signal.
"""

from src.strategy.candle_reject.config import CandleRejectConfig, ShortMode
from src.strategy.candle_reject.preprocessor import preprocess
from src.strategy.candle_reject.signal import generate_signals
from src.strategy.candle_reject.strategy import CandleRejectStrategy

__all__ = [
    "CandleRejectConfig",
    "CandleRejectStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
