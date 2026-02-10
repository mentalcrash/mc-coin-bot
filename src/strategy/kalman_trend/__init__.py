"""Adaptive Kalman Trend Strategy.

Kalman Filter velocity → adaptive trend detection.
"""

from src.strategy.kalman_trend.config import KalmanTrendConfig, ShortMode
from src.strategy.kalman_trend.preprocessor import preprocess
from src.strategy.kalman_trend.signal import generate_signals
from src.strategy.kalman_trend.strategy import KalmanTrendStrategy

__all__ = [
    "KalmanTrendConfig",
    "KalmanTrendStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
