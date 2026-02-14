"""Dual Volatility Trend: YZ vs Parkinson vol ratio for trend detection."""

from src.strategy.dual_vol.config import DualVolConfig, ShortMode
from src.strategy.dual_vol.preprocessor import preprocess
from src.strategy.dual_vol.signal import generate_signals
from src.strategy.dual_vol.strategy import DualVolStrategy

__all__ = [
    "DualVolConfig",
    "DualVolStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
