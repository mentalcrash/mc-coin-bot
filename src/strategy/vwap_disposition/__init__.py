"""VWAP Disposition Momentum Strategy.

Rolling VWAP CGO → behavioral finance disposition effect.
"""

from src.strategy.vwap_disposition.config import ShortMode, VWAPDispositionConfig
from src.strategy.vwap_disposition.preprocessor import preprocess
from src.strategy.vwap_disposition.signal import generate_signals
from src.strategy.vwap_disposition.strategy import VWAPDispositionStrategy

__all__ = [
    "ShortMode",
    "VWAPDispositionConfig",
    "VWAPDispositionStrategy",
    "generate_signals",
    "preprocess",
]
