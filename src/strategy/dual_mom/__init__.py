"""Dual Momentum Strategy.

12H 최적화 횡단면 모멘텀 전략.
Per-symbol momentum signal + vol-target sizing.
Cross-sectional ranking은 IntraPodAllocator(DUAL_MOMENTUM)에서 수행.

Components:
    - DualMomConfig: 전략 설정 (Pydantic 모델)
    - preprocess: 지표 계산 함수 (벡터화)
    - generate_signals: 시그널 생성 함수
    - DualMomStrategy: 전략 클래스 (BaseStrategy 상속)
"""

from src.strategy.dual_mom.config import DualMomConfig
from src.strategy.dual_mom.preprocessor import preprocess
from src.strategy.dual_mom.signal import generate_signals
from src.strategy.dual_mom.strategy import DualMomStrategy

__all__ = [
    "DualMomConfig",
    "DualMomStrategy",
    "generate_signals",
    "preprocess",
]
