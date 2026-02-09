"""HMM Regime Strategy.

GaussianHMM 3-state (Bull/Bear/Sideways) regime classification.
"""

from src.strategy.hmm_regime.config import HMMRegimeConfig, ShortMode
from src.strategy.hmm_regime.preprocessor import preprocess
from src.strategy.hmm_regime.signal import generate_signals
from src.strategy.hmm_regime.strategy import HMMRegimeStrategy

__all__ = [
    "HMMRegimeConfig",
    "HMMRegimeStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
