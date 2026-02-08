"""Stochastic Momentum Hybrid Strategy.

Stochastic %K/%D crossover + SMA trend filter + ATR position sizing.

Example:
    >>> from src.strategy.stoch_mom import StochMomStrategy, StochMomConfig
    >>> strategy = StochMomStrategy()
    >>> processed_df, signals = strategy.run(ohlcv_df)
"""

from src.strategy.stoch_mom.config import ShortMode, StochMomConfig
from src.strategy.stoch_mom.preprocessor import preprocess
from src.strategy.stoch_mom.signal import generate_signals
from src.strategy.stoch_mom.strategy import StochMomStrategy

__all__ = [
    "ShortMode",
    "StochMomConfig",
    "StochMomStrategy",
    "generate_signals",
    "preprocess",
]
