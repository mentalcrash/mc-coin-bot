"""Trend Factor Multi-Horizon: 5-horizon risk-adjusted return 합산 momentum."""

from src.strategy.trend_factor_12h.config import ShortMode, TrendFactorConfig
from src.strategy.trend_factor_12h.preprocessor import preprocess
from src.strategy.trend_factor_12h.signal import generate_signals
from src.strategy.trend_factor_12h.strategy import TrendFactorStrategy

__all__ = [
    "ShortMode",
    "TrendFactorConfig",
    "TrendFactorStrategy",
    "generate_signals",
    "preprocess",
]
