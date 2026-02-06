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

from src.backtest.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from src.backtest.validation.levels import ValidationLevel
from src.backtest.validation.models import (
    MILESTONE_MAX_SHARPE_DECAY,
    MILESTONE_MIN_CONSISTENCY,
    MILESTONE_MIN_OOS_SHARPE,
    MULTI_CPCV_MAX_PBO,
    MULTI_DEFLATED_SHARPE_MIN,
    MULTI_QUICK_MAX_DEGRADATION,
    MULTI_QUICK_MIN_OOS_SHARPE,
    MULTI_WFA_MAX_SHARPE_DECAY,
    MULTI_WFA_MIN_CONSISTENCY,
    MULTI_WFA_MIN_OOS_SHARPE,
    QUICK_MAX_SHARPE_DECAY,
    QUICK_MIN_OOS_SHARPE,
    FoldResult,
    MonteCarloResult,
    SplitInfo,
    ValidationResult,
)
from src.backtest.validation.pbo import calculate_pbo, calculate_pbo_logit
from src.backtest.validation.report import generate_validation_report
from src.backtest.validation.splitters import (
    get_split_info_is_oos,
    split_cpcv,
    split_is_oos,
    split_multi_cpcv,
    split_multi_is_oos,
    split_multi_walk_forward,
    split_walk_forward,
)
from src.backtest.validation.validator import TieredValidator

__all__ = [
    "MILESTONE_MAX_SHARPE_DECAY",
    "MILESTONE_MIN_CONSISTENCY",
    "MILESTONE_MIN_OOS_SHARPE",
    "MULTI_CPCV_MAX_PBO",
    "MULTI_DEFLATED_SHARPE_MIN",
    "MULTI_QUICK_MAX_DEGRADATION",
    "MULTI_QUICK_MIN_OOS_SHARPE",
    "MULTI_WFA_MAX_SHARPE_DECAY",
    "MULTI_WFA_MIN_CONSISTENCY",
    "MULTI_WFA_MIN_OOS_SHARPE",
    "QUICK_MAX_SHARPE_DECAY",
    "QUICK_MIN_OOS_SHARPE",
    "FoldResult",
    "MonteCarloResult",
    "SplitInfo",
    "TieredValidator",
    "ValidationLevel",
    "ValidationResult",
    "calculate_pbo",
    "calculate_pbo_logit",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "generate_validation_report",
    "get_split_info_is_oos",
    "probabilistic_sharpe_ratio",
    "split_cpcv",
    "split_is_oos",
    "split_multi_cpcv",
    "split_multi_is_oos",
    "split_multi_walk_forward",
    "split_walk_forward",
]
