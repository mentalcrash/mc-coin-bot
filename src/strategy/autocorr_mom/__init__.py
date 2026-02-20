"""Autocorrelation Momentum: lag-1 자기상관 기반 모멘텀 레짐 감지."""

from src.strategy.autocorr_mom.config import AutocorrMomConfig, ShortMode
from src.strategy.autocorr_mom.preprocessor import preprocess
from src.strategy.autocorr_mom.signal import generate_signals
from src.strategy.autocorr_mom.strategy import AutocorrMomStrategy

__all__ = [
    "AutocorrMomConfig",
    "AutocorrMomStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
