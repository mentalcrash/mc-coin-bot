"""Hour Seasonality Strategy.

Per-hour rolling t-stat + volume confirm → intraday seasonal alpha.
"""

from src.strategy.hour_season.config import HourSeasonConfig, ShortMode
from src.strategy.hour_season.preprocessor import preprocess
from src.strategy.hour_season.signal import generate_signals
from src.strategy.hour_season.strategy import HourSeasonStrategy

__all__ = [
    "HourSeasonConfig",
    "HourSeasonStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
