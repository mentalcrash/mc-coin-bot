"""Stablecoin Composition Shift Strategy.

USDT/USDC 구성비 변화 기반 리스크 선호도 시그널.
USDT 비중 상승 = 리테일 risk-on, USDC 비중 상승 = 기관 cautious.

Components:
    - StabCompConfig: Pydantic frozen config
    - preprocess: USDT share 계산 + ROC
    - generate_signals: 7D/30D ROC 방향 기반 시그널
    - StabCompStrategy: @register("stab-comp")
"""

from src.strategy.stab_comp.config import StabCompConfig
from src.strategy.stab_comp.preprocessor import preprocess
from src.strategy.stab_comp.signal import generate_signals
from src.strategy.stab_comp.strategy import StabCompStrategy

__all__ = [
    "StabCompConfig",
    "StabCompStrategy",
    "generate_signals",
    "preprocess",
]
