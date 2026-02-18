"""Multi-Domain Score: 4차원 soft scoring 전략."""

from src.strategy.multi_domain_score.config import MultiDomainScoreConfig, ShortMode
from src.strategy.multi_domain_score.preprocessor import preprocess
from src.strategy.multi_domain_score.signal import generate_signals
from src.strategy.multi_domain_score.strategy import MultiDomainScoreStrategy

__all__ = [
    "MultiDomainScoreConfig",
    "MultiDomainScoreStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
