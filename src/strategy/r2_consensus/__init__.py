"""R2 Consensus Trend: 3-scale R^2 투표 consensus 추세 전략."""

from src.strategy.r2_consensus.config import R2ConsensusConfig, ShortMode
from src.strategy.r2_consensus.preprocessor import preprocess
from src.strategy.r2_consensus.signal import generate_signals
from src.strategy.r2_consensus.strategy import R2ConsensusStrategy

__all__ = [
    "R2ConsensusConfig",
    "R2ConsensusStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
