"""Carry-Sentiment Gate: FR carry premium + F&G sentiment gate."""

from src.strategy.carry_sent.config import CarrySentConfig, ShortMode
from src.strategy.carry_sent.preprocessor import preprocess
from src.strategy.carry_sent.signal import generate_signals
from src.strategy.carry_sent.strategy import CarrySentStrategy

__all__ = [
    "CarrySentConfig",
    "CarrySentStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
