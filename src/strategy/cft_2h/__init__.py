"""Conviction-Filtered Trend: 레짐 게이트 기반 모멘텀."""

from src.strategy.cft_2h.config import Cft2hConfig, ShortMode
from src.strategy.cft_2h.preprocessor import preprocess
from src.strategy.cft_2h.signal import generate_signals
from src.strategy.cft_2h.strategy import Cft2hStrategy

__all__ = [
    "Cft2hConfig",
    "Cft2hStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
