"""MAD-Channel Multi-Scale Trend: DC/KC/MAD 3채널 x 3스케일 앙상블 breakout 추세 추종."""

from src.strategy.mad_channel_trend.config import MadChannelTrendConfig, ShortMode
from src.strategy.mad_channel_trend.preprocessor import preprocess
from src.strategy.mad_channel_trend.signal import generate_signals
from src.strategy.mad_channel_trend.strategy import MadChannelTrendStrategy

__all__ = [
    "MadChannelTrendConfig",
    "MadChannelTrendStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
