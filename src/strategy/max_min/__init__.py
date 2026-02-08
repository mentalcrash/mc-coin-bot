"""MAX/MIN Combined Strategy.

신고가 매수(trend) + 신저가 매수(mean reversion) 복합 전략.
"""

from src.strategy.max_min.config import MaxMinConfig
from src.strategy.max_min.preprocessor import preprocess
from src.strategy.max_min.signal import generate_signals
from src.strategy.max_min.strategy import MaxMinStrategy

__all__ = [
    "MaxMinConfig",
    "MaxMinStrategy",
    "generate_signals",
    "preprocess",
]
