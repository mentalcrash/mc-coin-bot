"""Efficiency Breakout: Kaufman ER threshold breakout for trend detection."""

from src.strategy.eff_brk.config import EffBrkConfig, ShortMode
from src.strategy.eff_brk.preprocessor import preprocess
from src.strategy.eff_brk.signal import generate_signals
from src.strategy.eff_brk.strategy import EffBrkStrategy

__all__ = [
    "EffBrkConfig",
    "EffBrkStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
