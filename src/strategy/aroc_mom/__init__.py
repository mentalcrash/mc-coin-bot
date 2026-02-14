"""Adaptive ROC Momentum: volatility-adaptive ROC lookback momentum."""

from src.strategy.aroc_mom.config import ArocMomConfig, ShortMode
from src.strategy.aroc_mom.preprocessor import preprocess
from src.strategy.aroc_mom.signal import generate_signals
from src.strategy.aroc_mom.strategy import ArocMomStrategy

__all__ = [
    "ArocMomConfig",
    "ArocMomStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
