"""Multi-Factor Ensemble Strategy.

3개의 직교 팩터(모멘텀, 거래량 충격, 역변동성)를 균등 가중 결합합니다.
"""

from src.strategy.multi_factor.config import MultiFactorConfig
from src.strategy.multi_factor.preprocessor import preprocess
from src.strategy.multi_factor.signal import generate_signals
from src.strategy.multi_factor.strategy import MultiFactorStrategy
from src.strategy.tsmom.config import ShortMode

__all__ = [
    "MultiFactorConfig",
    "MultiFactorStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
