"""VWAP Trend Crossover: rolling VWAP crossover detects participant entry price shifts."""

from src.strategy.vwap_trend_cross.config import ShortMode, VwapTrendCrossConfig
from src.strategy.vwap_trend_cross.preprocessor import preprocess
from src.strategy.vwap_trend_cross.signal import generate_signals
from src.strategy.vwap_trend_cross.strategy import VwapTrendCrossStrategy

__all__ = [
    "ShortMode",
    "VwapTrendCrossConfig",
    "VwapTrendCrossStrategy",
    "generate_signals",
    "preprocess",
]
