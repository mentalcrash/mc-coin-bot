"""Copula Pairs Trading Strategy.

Engle-Granger cointegration -> spread -> z-score -> mean-reversion signals.
Full copula fitting is deferred to a later phase.
"""

from src.strategy.copula_pairs.config import CopulaPairsConfig
from src.strategy.copula_pairs.preprocessor import preprocess
from src.strategy.copula_pairs.signal import generate_signals
from src.strategy.copula_pairs.strategy import CopulaPairsStrategy
from src.strategy.tsmom.config import ShortMode

__all__ = [
    "CopulaPairsConfig",
    "CopulaPairsStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
