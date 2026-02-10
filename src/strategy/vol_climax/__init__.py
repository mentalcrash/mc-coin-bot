"""Volume Climax Reversal Strategy.

Extreme volume spikes (climax) = capitulation/euphoria -> reversal.
"""

from src.strategy.vol_climax.config import ShortMode, VolClimaxConfig
from src.strategy.vol_climax.preprocessor import preprocess
from src.strategy.vol_climax.signal import generate_signals
from src.strategy.vol_climax.strategy import VolClimaxStrategy

__all__ = [
    "ShortMode",
    "VolClimaxConfig",
    "VolClimaxStrategy",
    "generate_signals",
    "preprocess",
]
