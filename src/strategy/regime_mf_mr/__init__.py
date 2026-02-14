"""Regime-Gated Multi-Factor MR: Ranging 레짐 전용 멀티팩터 평균회귀."""

from src.strategy.regime_mf_mr.config import RegimeMfMrConfig, ShortMode
from src.strategy.regime_mf_mr.preprocessor import preprocess
from src.strategy.regime_mf_mr.signal import generate_signals
from src.strategy.regime_mf_mr.strategy import RegimeMfMrStrategy

__all__ = [
    "RegimeMfMrConfig",
    "RegimeMfMrStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
