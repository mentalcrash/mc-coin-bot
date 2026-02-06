"""Strategy module for trading strategies.

이 모듈은 트레이딩 전략의 기반 클래스와 타입 정의를 제공합니다.
모든 전략은 BaseStrategy를 상속받아 구현됩니다.

Registry Pattern:
    전략은 @register() 데코레이터로 등록되며, get_strategy()로 조회합니다.
    이를 통해 CLI와 전략 간의 결합도를 제거하여 OCP를 준수합니다.

Example:
    >>> from src.strategy import BaseStrategy, StrategySignals, Direction
    >>> from src.strategy import get_strategy, list_strategies
    >>>
    >>> # 이름으로 전략 조회
    >>> strategy_class = get_strategy("tsmom")
    >>> strategy = strategy_class()
    >>>
    >>> # 등록된 전략 목록
    >>> print(list_strategies())  # ['adaptive-breakout', 'tsmom']
"""

# pyright: reportUnusedImport=false

# 전략 자동 등록 (import 시 @register 데코레이터 실행)
# NOTE: 각 전략 모듈이 임포트될 때 Registry에 등록됨
import src.strategy.bb_rsi  # 전략 등록 side effect
import src.strategy.breakout  # 전략 등록 side effect
import src.strategy.donchian  # 전략 등록 side effect
import src.strategy.tsmom  # 전략 등록 side effect
from src.strategy.base import BaseStrategy
from src.strategy.registry import (
    get_strategy,
    is_registered,
    list_strategies,
    register,
)
from src.strategy.types import (
    DEFAULT_OHLCV_COLUMNS,
    Direction,
    SignalType,
    StrategySignals,
)

__all__ = [
    "DEFAULT_OHLCV_COLUMNS",
    # Base & Types
    "BaseStrategy",
    "Direction",
    "SignalType",
    "StrategySignals",
    # Registry
    "get_strategy",
    "is_registered",
    "list_strategies",
    "register",
]
