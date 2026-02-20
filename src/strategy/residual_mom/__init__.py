"""Residual Momentum: 시장 factor 제거 잔차의 모멘텀으로 자산 고유 alpha 포착."""

from src.strategy.residual_mom.config import ResidualMomConfig, ShortMode
from src.strategy.residual_mom.preprocessor import preprocess
from src.strategy.residual_mom.signal import generate_signals
from src.strategy.residual_mom.strategy import ResidualMomStrategy

__all__ = [
    "ResidualMomConfig",
    "ResidualMomStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
