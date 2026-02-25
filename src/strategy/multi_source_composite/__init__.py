"""Multi-Source Directional Composite: 3개 직교 소스 majority vote 전략."""

from src.strategy.multi_source_composite.config import MultiSourceCompositeConfig, ShortMode
from src.strategy.multi_source_composite.preprocessor import preprocess
from src.strategy.multi_source_composite.signal import generate_signals
from src.strategy.multi_source_composite.strategy import MultiSourceCompositeStrategy

__all__ = [
    "MultiSourceCompositeConfig",
    "MultiSourceCompositeStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
