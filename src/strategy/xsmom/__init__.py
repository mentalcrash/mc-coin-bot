"""XSMOM (Cross-Sectional Momentum) Strategy.

이 모듈은 횡단면 모멘텀 전략을 구현합니다.
코인별 rolling return + vol-target sizing으로
cross-sectional ranking은 멀티에셋 백테스트 레벨에서 수행됩니다.

Pure XSMOM + Vol Target 구현:
    1. rolling_return = lookback 기간 수익률
    2. vol_scalar = vol_target / realized_vol
    3. direction = sign(rolling_return)
    4. strength = direction * vol_scalar
    5. holding_period로 시그널 리밸런싱 주기 제어

Components:
    - XSMOMConfig: 전략 설정 (Pydantic 모델)
    - preprocess: 지표 계산 함수 (벡터화)
    - generate_signals: 시그널 생성 함수
    - XSMOMStrategy: 전략 클래스 (BaseStrategy 상속)

Example:
    >>> from src.strategy.xsmom import XSMOMStrategy, XSMOMConfig
    >>> config = XSMOMConfig(lookback=21, vol_target=0.35)
    >>> strategy = XSMOMStrategy(config)
    >>> processed_df, signals = strategy.run(ohlcv_df)
"""

from src.strategy.xsmom.config import XSMOMConfig
from src.strategy.xsmom.preprocessor import preprocess
from src.strategy.xsmom.signal import generate_signals
from src.strategy.xsmom.strategy import XSMOMStrategy

__all__ = [
    "XSMOMConfig",
    "XSMOMStrategy",
    "generate_signals",
    "preprocess",
]
