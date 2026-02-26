"""Triple-Channel Multi-Scale Trend: 3종 채널 x 3스케일 앙상블 breakout 추세 추종."""

from src.strategy.tri_channel_trend.config import ShortMode, TriChannelTrendConfig
from src.strategy.tri_channel_trend.preprocessor import preprocess
from src.strategy.tri_channel_trend.signal import generate_signals
from src.strategy.tri_channel_trend.strategy import TriChannelTrendStrategy

__all__ = [
    "ShortMode",
    "TriChannelTrendConfig",
    "TriChannelTrendStrategy",
    "generate_signals",
    "preprocess",
]
