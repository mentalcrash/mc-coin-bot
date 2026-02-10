"""Permutation Entropy Momentum Strategy.

Low Permutation Entropy = orderly market -> high conviction momentum.
High PE = noise -> reduced/zero position.
"""

from src.strategy.perm_entropy_mom.config import PermEntropyMomConfig, ShortMode
from src.strategy.perm_entropy_mom.preprocessor import preprocess
from src.strategy.perm_entropy_mom.signal import generate_signals
from src.strategy.perm_entropy_mom.strategy import PermEntropyMomStrategy

__all__ = [
    "PermEntropyMomConfig",
    "PermEntropyMomStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
