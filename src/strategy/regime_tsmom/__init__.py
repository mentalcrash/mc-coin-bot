"""Regime-Adaptive TSMOM Strategy.

TSMOM을 레짐 정보로 강화한 전략입니다.
RegimeDetector를 활용하여 시장 레짐에 따라 vol_target과
leverage scale을 적응적으로 조절합니다.

Components:
    - RegimeTSMOMConfig: 전략 설정 (TSMOM + 레짐 파라미터)
    - preprocess: 레짐 + TSMOM 지표 계산
    - generate_signals: 레짐 적응적 시그널 생성
    - RegimeTSMOMStrategy: 전략 클래스 (@register("regime-tsmom"))

Example:
    >>> from src.strategy.regime_tsmom import RegimeTSMOMStrategy
    >>> strategy = RegimeTSMOMStrategy()
    >>> processed_df, signals = strategy.run(ohlcv_df)
"""

from src.strategy.regime_tsmom.config import RegimeTSMOMConfig
from src.strategy.regime_tsmom.preprocessor import preprocess
from src.strategy.regime_tsmom.signal import generate_signals
from src.strategy.regime_tsmom.strategy import RegimeTSMOMStrategy

__all__ = [
    "RegimeTSMOMConfig",
    "RegimeTSMOMStrategy",
    "generate_signals",
    "preprocess",
]
