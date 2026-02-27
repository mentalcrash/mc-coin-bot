"""Wavelet-Channel 8H: DWT denoised 3-channel x 3-scale ensemble breakout 추세 추종."""

from src.strategy.wavelet_channel_8h.config import ShortMode, WaveletChannel8hConfig
from src.strategy.wavelet_channel_8h.preprocessor import preprocess
from src.strategy.wavelet_channel_8h.signal import generate_signals
from src.strategy.wavelet_channel_8h.strategy import WaveletChannel8hStrategy

__all__ = [
    "ShortMode",
    "WaveletChannel8hConfig",
    "WaveletChannel8hStrategy",
    "generate_signals",
    "preprocess",
]
