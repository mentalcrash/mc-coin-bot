"""VW-TSMOM (Volume-Weighted Time Series Momentum) Pure Strategy.

이 모듈은 거래량 가중 수익률만을 사용하는 순수 VW-TSMOM 전략을 구현합니다.
기존 tsmom 전략의 간소화 변형으로, VW returns에만 집중합니다.

Pure VW-TSMOM 구현:
    1. vw_returns = 거래량 가중 수익률 (lookback 기간)
    2. vol_scalar = vol_target / realized_vol
    3. direction = sign(vw_returns)
    4. strength = direction * vol_scalar

Components:
    - VWTSMOMConfig: 전략 설정 (Pydantic 모델)
    - preprocess: 지표 계산 함수 (벡터화)
    - generate_signals: 시그널 생성 함수
    - VWTSMOMStrategy: 전략 클래스 (BaseStrategy 상속)

Example:
    >>> from src.strategy.vw_tsmom import VWTSMOMStrategy, VWTSMOMConfig
    >>> config = VWTSMOMConfig(lookback=21, vol_target=0.35)
    >>> strategy = VWTSMOMStrategy(config)
    >>> processed_df, signals = strategy.run(ohlcv_df)
"""

from src.strategy.vw_tsmom.config import VWTSMOMConfig
from src.strategy.vw_tsmom.preprocessor import preprocess
from src.strategy.vw_tsmom.signal import generate_signals
from src.strategy.vw_tsmom.strategy import VWTSMOMStrategy

__all__ = [
    "VWTSMOMConfig",
    "VWTSMOMStrategy",
    "generate_signals",
    "preprocess",
]
