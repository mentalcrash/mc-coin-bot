"""Macro-Gated Patient Trend (4H): Macro z-score gate + 3-scale Donchian breakout."""

from src.strategy.macro_patience_4h.config import MacroPatience4hConfig, ShortMode
from src.strategy.macro_patience_4h.preprocessor import preprocess
from src.strategy.macro_patience_4h.signal import generate_signals
from src.strategy.macro_patience_4h.strategy import MacroPatience4hStrategy

__all__ = [
    "MacroPatience4hConfig",
    "MacroPatience4hStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
