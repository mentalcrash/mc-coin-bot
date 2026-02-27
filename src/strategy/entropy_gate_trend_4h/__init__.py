"""Entropy-Gate Trend 4H: Sample/Permutation Entropy 기반 게이팅 + 3-scale Donchian 추세 추종."""

from src.strategy.entropy_gate_trend_4h.config import EntropyGateTrend4hConfig, ShortMode
from src.strategy.entropy_gate_trend_4h.preprocessor import preprocess
from src.strategy.entropy_gate_trend_4h.signal import generate_signals
from src.strategy.entropy_gate_trend_4h.strategy import EntropyGateTrend4hStrategy

__all__ = [
    "EntropyGateTrend4hConfig",
    "EntropyGateTrend4hStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
