"""Anchored Trend-Following 3H: Anchor-Mom의 3H TF 적응."""

from src.strategy.atf_3h.config import Atf3hConfig, ShortMode
from src.strategy.atf_3h.preprocessor import preprocess
from src.strategy.atf_3h.signal import generate_signals
from src.strategy.atf_3h.strategy import Atf3hStrategy

__all__ = [
    "Atf3hConfig",
    "Atf3hStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
