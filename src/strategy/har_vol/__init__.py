"""HAR Volatility Overlay Strategy.

HAR-RV (Heterogeneous Autoregressive Realized Volatility) 모델을 사용하여
변동성을 예측하고, vol surprise(realized - forecast)로 매매하는 전략입니다.
"""

from src.strategy.har_vol.config import HARVolConfig
from src.strategy.har_vol.preprocessor import preprocess
from src.strategy.har_vol.signal import generate_signals
from src.strategy.har_vol.strategy import HARVolStrategy
from src.strategy.tsmom.config import ShortMode

__all__ = [
    "HARVolConfig",
    "HARVolStrategy",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
