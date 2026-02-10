"""Variance Ratio Regime Strategy.

Lo-MacKinlay VR test → trending/MR regime detection.
"""

from src.strategy.vr_regime.config import ShortMode, VRRegimeConfig
from src.strategy.vr_regime.preprocessor import preprocess
from src.strategy.vr_regime.signal import generate_signals
from src.strategy.vr_regime.strategy import VRRegimeStrategy

__all__ = [
    "ShortMode",
    "VRRegimeConfig",
    "VRRegimeStrategy",
    "generate_signals",
    "preprocess",
]
