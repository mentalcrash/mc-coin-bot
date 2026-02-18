"""OI-Price Divergence Strategy.

OI-가격 괴리 + Funding Rate z-score 기반 숏스퀴즈/롱청산 감지.
BTC/ETH 전용 (derivatives data scope).

Components:
    - OiDivergeConfig: Pydantic frozen config
    - preprocess: OI divergence + FR zscore + vol 지표 계산
    - generate_signals: 스퀴즈/청산 시그널
    - OiDivergeStrategy: @register("oi-diverge")
"""

from src.strategy.oi_diverge.config import OiDivergeConfig
from src.strategy.oi_diverge.preprocessor import preprocess
from src.strategy.oi_diverge.signal import generate_signals
from src.strategy.oi_diverge.strategy import OiDivergeStrategy

__all__ = [
    "OiDivergeConfig",
    "OiDivergeStrategy",
    "generate_signals",
    "preprocess",
]
