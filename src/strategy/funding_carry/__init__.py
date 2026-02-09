"""Funding Rate Carry Strategy.

Positive funding rate -> short (receive carry), Negative -> long.
"""

from src.strategy.funding_carry.config import FundingCarryConfig
from src.strategy.funding_carry.preprocessor import preprocess
from src.strategy.funding_carry.signal import generate_signals
from src.strategy.funding_carry.strategy import FundingCarryStrategy

__all__ = [
    "FundingCarryConfig",
    "FundingCarryStrategy",
    "generate_signals",
    "preprocess",
]
