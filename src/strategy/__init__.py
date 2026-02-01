"""Strategy module for trading strategies.

이 모듈은 트레이딩 전략의 기반 클래스와 타입 정의를 제공합니다.
모든 전략은 BaseStrategy를 상속받아 구현됩니다.

Example:
    >>> from src.strategy import BaseStrategy, StrategySignals, Direction
    >>> from src.strategy.tsmom import TSMOMStrategy
"""

from src.strategy.base import BaseStrategy
from src.strategy.types import (
    DEFAULT_OHLCV_COLUMNS,
    Direction,
    SignalType,
    StrategySignals,
)

__all__ = [
    "DEFAULT_OHLCV_COLUMNS",
    "BaseStrategy",
    "Direction",
    "SignalType",
    "StrategySignals",
]
