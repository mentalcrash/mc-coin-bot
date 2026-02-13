"""Trend Quality Momentum: R^2-weighted momentum with trend quality conviction."""

from src.strategy.trend_quality_mom.config import ShortMode, TrendQualityMomConfig
from src.strategy.trend_quality_mom.preprocessor import preprocess
from src.strategy.trend_quality_mom.signal import generate_signals
from src.strategy.trend_quality_mom.strategy import TrendQualityMomStrategy

__all__ = [
    "ShortMode",
    "TrendQualityMomConfig",
    "TrendQualityMomStrategy",
    "generate_signals",
    "preprocess",
]
