"""Capital Gains Overhang Momentum: CGO 기반 disposition effect 모멘텀 전략."""

from src.strategy.cgo_mom.config import CgoMomConfig, ShortMode
from src.strategy.cgo_mom.preprocessor import preprocess
from src.strategy.cgo_mom.signal import generate_signals
from src.strategy.cgo_mom.strategy import CgoMomStrategy

__all__ = [
    "CgoMomConfig",
    "CgoMomStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
