"""Acceleration-Conviction Momentum: 가속도 x body conviction 추세 지속 포착."""

from src.strategy.accel_conv.config import AccelConvConfig, ShortMode
from src.strategy.accel_conv.preprocessor import preprocess
from src.strategy.accel_conv.signal import generate_signals
from src.strategy.accel_conv.strategy import AccelConvStrategy

__all__ = [
    "AccelConvConfig",
    "AccelConvStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
