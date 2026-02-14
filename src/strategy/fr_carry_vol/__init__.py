"""FR Carry Vol: Funding rate contrarian carry with volatility conditioning."""

from src.strategy.fr_carry_vol.config import FRCarryVolConfig, ShortMode
from src.strategy.fr_carry_vol.preprocessor import preprocess
from src.strategy.fr_carry_vol.signal import generate_signals
from src.strategy.fr_carry_vol.strategy import FRCarryVolStrategy

__all__ = [
    "FRCarryVolConfig",
    "FRCarryVolStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
