"""Adaptive Breakout Strategy module.

Donchian Channel 기반 돌파 전략으로, ATR을 활용하여
변동성에 적응하는 임계값을 사용합니다.

Example:
    >>> from src.strategy.breakout import AdaptiveBreakoutStrategy, AdaptiveBreakoutConfig
    >>>
    >>> # 기본 설정으로 생성
    >>> strategy = AdaptiveBreakoutStrategy()
    >>>
    >>> # 보수적 설정으로 생성
    >>> strategy = AdaptiveBreakoutStrategy.conservative()
"""

from src.strategy.breakout.config import AdaptiveBreakoutConfig, ShortMode
from src.strategy.breakout.strategy import AdaptiveBreakoutStrategy

__all__ = [
    "AdaptiveBreakoutConfig",
    "AdaptiveBreakoutStrategy",
    "ShortMode",
]
