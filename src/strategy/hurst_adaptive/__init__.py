"""Hurst-Adaptive Strategy.

Hurst Exponent + Efficiency Ratio 기반 레짐 감지 전략.
- 추세 구간 (H > 0.55, ER > 0.40): 모멘텀 추종
- 횡보 구간 (H < 0.45, ER < 0.40): 평균회귀
- Dead zone (0.45~0.55): 시그널 없음

Components:
    - HurstAdaptiveConfig: Pydantic frozen config
    - preprocess: hurst + ER + vol 지표 계산
    - generate_signals: 레짐 스위칭 시그널
    - HurstAdaptiveStrategy: @register("hurst-adaptive")
"""

from src.strategy.hurst_adaptive.config import HurstAdaptiveConfig
from src.strategy.hurst_adaptive.preprocessor import preprocess
from src.strategy.hurst_adaptive.signal import generate_signals
from src.strategy.hurst_adaptive.strategy import HurstAdaptiveStrategy

__all__ = [
    "HurstAdaptiveConfig",
    "HurstAdaptiveStrategy",
    "generate_signals",
    "preprocess",
]
