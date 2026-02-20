"""Conviction Trend Composite: 가격 모멘텀 + OBV/RV conviction + 레짐 적응형."""

from src.strategy.conviction_trend_composite.config import (
    ConvictionTrendCompositeConfig,
    ShortMode,
)
from src.strategy.conviction_trend_composite.preprocessor import preprocess
from src.strategy.conviction_trend_composite.signal import generate_signals
from src.strategy.conviction_trend_composite.strategy import ConvictionTrendCompositeStrategy

__all__ = [
    "ConvictionTrendCompositeConfig",
    "ConvictionTrendCompositeStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
