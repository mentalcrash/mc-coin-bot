"""Autocorrelation Regime-Adaptive Strategy.

Rolling autocorrelation → trending/MR regime detection → auto-switch.
"""

from src.strategy.ac_regime.config import ACRegimeConfig, ShortMode
from src.strategy.ac_regime.preprocessor import preprocess
from src.strategy.ac_regime.signal import generate_signals
from src.strategy.ac_regime.strategy import ACRegimeStrategy

__all__ = [
    "ACRegimeConfig",
    "ACRegimeStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
