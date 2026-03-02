"""Carry-Regime Trend: 12H multi-scale EMA trend + FR percentile adaptive exit."""

from src.strategy.carry_regime_12h.config import CarryRegimeConfig, ShortMode
from src.strategy.carry_regime_12h.preprocessor import preprocess
from src.strategy.carry_regime_12h.signal import generate_signals
from src.strategy.carry_regime_12h.strategy import CarryRegimeStrategy

__all__ = [
    "CarryRegimeConfig",
    "CarryRegimeStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
