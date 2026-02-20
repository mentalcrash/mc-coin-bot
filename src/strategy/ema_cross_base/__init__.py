"""EMA Cross Base — 순수 20/100 EMA 크로스오버 베이스라인 전략."""

from src.strategy.ema_cross_base.config import EmaCrossBaseConfig, ShortMode
from src.strategy.ema_cross_base.preprocessor import preprocess
from src.strategy.ema_cross_base.signal import generate_signals
from src.strategy.ema_cross_base.strategy import EmaCrossBaseStrategy

__all__ = [
    "EmaCrossBaseConfig",
    "EmaCrossBaseStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
