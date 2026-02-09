"""Hurst/ER Regime Strategy.

Efficiency Ratio + R/S Hurst exponent → trending/mean-reverting regime classification.
"""

from src.strategy.hurst_regime.config import HurstRegimeConfig, ShortMode
from src.strategy.hurst_regime.preprocessor import preprocess
from src.strategy.hurst_regime.signal import generate_signals
from src.strategy.hurst_regime.strategy import HurstRegimeStrategy

__all__ = [
    "HurstRegimeConfig",
    "HurstRegimeStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
