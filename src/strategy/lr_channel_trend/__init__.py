"""LR-Channel Multi-Scale Trend: 선형회귀 잔차 채널 3스케일 consensus 추세 추종."""

from src.strategy.lr_channel_trend.config import LrChannelTrendConfig, ShortMode
from src.strategy.lr_channel_trend.preprocessor import preprocess
from src.strategy.lr_channel_trend.signal import generate_signals
from src.strategy.lr_channel_trend.strategy import LrChannelTrendStrategy

__all__ = [
    "LrChannelTrendConfig",
    "LrChannelTrendStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
