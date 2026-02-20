"""Trend Persistence Score: 수익률 부호 일관성 기반 추세 품질 측정."""

from src.strategy.trend_persist.config import ShortMode, TrendPersistConfig
from src.strategy.trend_persist.preprocessor import preprocess
from src.strategy.trend_persist.signal import generate_signals
from src.strategy.trend_persist.strategy import TrendPersistStrategy

__all__ = [
    "ShortMode",
    "TrendPersistConfig",
    "TrendPersistStrategy",
    "generate_signals",
    "preprocess",
]
