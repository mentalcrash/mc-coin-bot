"""ER Trend: Multi-lookback Signed ER 가중 합성 추세 품질 모멘텀."""

from src.strategy.er_trend.config import ErTrendConfig, ShortMode
from src.strategy.er_trend.preprocessor import preprocess
from src.strategy.er_trend.signal import generate_signals
from src.strategy.er_trend.strategy import ErTrendStrategy

__all__ = [
    "ErTrendConfig",
    "ErTrendStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
