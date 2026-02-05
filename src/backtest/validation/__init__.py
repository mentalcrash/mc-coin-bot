"""Tiered Validation module.

전략 검증을 위한 단계별 검증 시스템을 제공합니다.

Usage:
    >>> from src.backtest.validation import (
    ...     ValidationLevel,
    ...     TieredValidator,
    ...     ValidationResult,
    ... )
    >>>
    >>> validator = TieredValidator()
    >>> result = validator.validate(
    ...     level=ValidationLevel.QUICK,
    ...     data=market_data,
    ...     strategy=strategy,
    ...     portfolio=portfolio,
    ... )
    >>> print(result.verdict)
"""

from src.backtest.validation.levels import ValidationLevel
from src.backtest.validation.models import (
    MILESTONE_MAX_SHARPE_DECAY,
    MILESTONE_MIN_CONSISTENCY,
    MILESTONE_MIN_OOS_SHARPE,
    QUICK_MAX_SHARPE_DECAY,
    QUICK_MIN_OOS_SHARPE,
    FoldResult,
    MonteCarloResult,
    SplitInfo,
    ValidationResult,
)
from src.backtest.validation.splitters import (
    get_split_info_is_oos,
    split_cpcv,
    split_is_oos,
    split_walk_forward,
)
from src.backtest.validation.validator import TieredValidator

__all__ = [
    "MILESTONE_MAX_SHARPE_DECAY",
    "MILESTONE_MIN_CONSISTENCY",
    "MILESTONE_MIN_OOS_SHARPE",
    "QUICK_MAX_SHARPE_DECAY",
    "QUICK_MIN_OOS_SHARPE",
    "FoldResult",
    "MonteCarloResult",
    "SplitInfo",
    "TieredValidator",
    "ValidationLevel",
    "ValidationResult",
    "get_split_info_is_oos",
    "split_cpcv",
    "split_is_oos",
    "split_walk_forward",
]
