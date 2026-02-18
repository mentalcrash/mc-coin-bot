"""Trend Efficiency Scorer: ER 기반 추세 품질 + 다중 ROC 합의."""

from src.strategy.trend_eff_score.config import ShortMode, TrendEffScoreConfig
from src.strategy.trend_eff_score.preprocessor import preprocess
from src.strategy.trend_eff_score.signal import generate_signals
from src.strategy.trend_eff_score.strategy import TrendEffScoreStrategy

__all__ = [
    "ShortMode",
    "TrendEffScoreConfig",
    "TrendEffScoreStrategy",
    "generate_signals",
    "preprocess",
]
