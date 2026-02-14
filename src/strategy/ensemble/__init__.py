"""Ensemble Strategy.

여러 서브 전략의 시그널을 집계하는 메타 전략.

Example:
    >>> from src.strategy.ensemble import EnsembleStrategy, EnsembleConfig
    >>> from src.strategy.ensemble.config import SubStrategySpec
    >>> config = EnsembleConfig(
    ...     strategies=(SubStrategySpec(name="tsmom"), SubStrategySpec(name="donchian-ensemble")),
    ... )
    >>> strategy = EnsembleStrategy(config)
    >>> processed_df, signals = strategy.run(ohlcv_df)
"""

from src.strategy.ensemble.config import (
    AggregationMethod,
    EnsembleConfig,
    ShortMode,
    SubStrategySpec,
)
from src.strategy.ensemble.strategy import EnsembleStrategy

__all__ = [
    "AggregationMethod",
    "EnsembleConfig",
    "EnsembleStrategy",
    "ShortMode",
    "SubStrategySpec",
]
