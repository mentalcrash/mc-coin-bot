"""Range Compression Breakout Strategy.

NR 패턴 + range ratio squeeze → breakout direction following.
"""

from src.strategy.range_squeeze.config import RangeSqueezeConfig, ShortMode
from src.strategy.range_squeeze.preprocessor import preprocess
from src.strategy.range_squeeze.signal import generate_signals
from src.strategy.range_squeeze.strategy import RangeSqueezeStrategy

__all__ = [
    "RangeSqueezeConfig",
    "RangeSqueezeStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
