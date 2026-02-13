"""Candle Conviction Momentum: body ratio 기반 rolling conviction 추세추종."""

from src.strategy.candle_conv_mom.config import CandleConvMomConfig, ShortMode
from src.strategy.candle_conv_mom.preprocessor import preprocess
from src.strategy.candle_conv_mom.signal import generate_signals
from src.strategy.candle_conv_mom.strategy import CandleConvMomStrategy

__all__ = [
    "CandleConvMomConfig",
    "CandleConvMomStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
