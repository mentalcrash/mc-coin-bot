"""Entropy Regime Switch Strategy.

Shannon Entropy → market predictability → trend-following filter.
"""

from src.strategy.entropy_switch.config import EntropySwitchConfig, ShortMode
from src.strategy.entropy_switch.preprocessor import preprocess
from src.strategy.entropy_switch.signal import generate_signals
from src.strategy.entropy_switch.strategy import EntropySwitchStrategy

__all__ = [
    "EntropySwitchConfig",
    "EntropySwitchStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
