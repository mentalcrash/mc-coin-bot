"""TTM Squeeze Strategy.

BB가 KC 안으로 수축 후 해제 시 momentum 방향 진입.

Example:
    >>> from src.strategy.ttm_squeeze import TtmSqueezeStrategy, TtmSqueezeConfig
    >>> strategy = TtmSqueezeStrategy()
    >>> processed_df, signals = strategy.run(ohlcv_df)
"""

from src.strategy.ttm_squeeze.config import ShortMode, TtmSqueezeConfig
from src.strategy.ttm_squeeze.preprocessor import preprocess
from src.strategy.ttm_squeeze.signal import generate_signals
from src.strategy.ttm_squeeze.strategy import TtmSqueezeStrategy

__all__ = [
    "ShortMode",
    "TtmSqueezeConfig",
    "TtmSqueezeStrategy",
    "generate_signals",
    "preprocess",
]
