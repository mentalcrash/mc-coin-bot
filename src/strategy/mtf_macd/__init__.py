"""MTF MACD Strategy.

MACD(12,26,9) 추세 필터 + crossover 진입 전략.

Example:
    >>> from src.strategy.mtf_macd import MtfMacdStrategy, MtfMacdConfig
    >>> strategy = MtfMacdStrategy()
    >>> processed_df, signals = strategy.run(ohlcv_df)
"""

from src.strategy.mtf_macd.config import MtfMacdConfig, ShortMode
from src.strategy.mtf_macd.strategy import MtfMacdStrategy

__all__ = [
    "MtfMacdConfig",
    "MtfMacdStrategy",
    "ShortMode",
]
