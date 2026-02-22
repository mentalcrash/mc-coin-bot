"""Multi-Source: 다중 데이터 소스 결합 전략."""

from src.strategy.multi_source.config import (
    MultiSourceConfig,
    ShortMode,
    SignalCombineMethod,
    SubSignalSpec,
    SubSignalTransform,
)
from src.strategy.multi_source.preprocessor import preprocess
from src.strategy.multi_source.signal import generate_signals
from src.strategy.multi_source.strategy import MultiSourceStrategy

__all__ = [
    "MultiSourceConfig",
    "MultiSourceStrategy",
    "ShortMode",
    "SignalCombineMethod",
    "SubSignalSpec",
    "SubSignalTransform",
    "generate_signals",
    "preprocess",
]
