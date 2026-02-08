"""KAMA Trend Following Strategy.

Kaufman Adaptive Moving Average 기반 추세 추종 전략입니다.
"""

from src.strategy.kama.config import KAMAConfig, ShortMode
from src.strategy.kama.preprocessor import preprocess
from src.strategy.kama.signal import generate_signals
from src.strategy.kama.strategy import KAMAStrategy

__all__ = [
    "KAMAConfig",
    "KAMAStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
