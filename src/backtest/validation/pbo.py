"""Probability of Backtest Overfitting (PBO) implementation.

Marcos Lopez de Prado (2019) CPCV 기반 과적합 확률 추정.

CPCV 결과에서 IS 최적 전략이 OOS에서도 최적인지 확률을 계산합니다.
PBO < 0.5이면 과적합 확률이 낮습니다.

Reference:
    Lopez de Prado, M. (2018).
    "Advances in Financial Machine Learning", Chapter 11-12.
"""

from __future__ import annotations

import numpy as np


def calculate_pbo(
    is_sharpes: list[float],
    oos_sharpes: list[float],
) -> float:
    """Probability of Backtest Overfitting (PBO) 계산.

    각 CPCV fold에서 IS 성과와 OOS 성과를 비교하여
    IS에서의 최적화가 OOS에서도 유효한지 추정합니다.

    방법:
        IS Sharpe의 랭크와 OOS Sharpe의 랭크를 비교.
        IS에서 좋은 성과가 OOS에서도 좋으면 PBO 낮음.
        IS에서 좋은 성과가 OOS에서 나쁘면 PBO 높음.

    Args:
        is_sharpes: CPCV fold별 In-Sample Sharpe 목록
        oos_sharpes: CPCV fold별 Out-of-Sample Sharpe 목록

    Returns:
        PBO (0~1). 0.5 미만이면 과적합 확률 낮음.

    Raises:
        ValueError: IS/OOS 길이가 다르거나 비어있는 경우
    """
    if len(is_sharpes) != len(oos_sharpes):
        msg = f"IS and OOS sharpes must have same length: {len(is_sharpes)} vs {len(oos_sharpes)}"
        raise ValueError(msg)

    n = len(is_sharpes)
    min_folds = 2
    if n < min_folds:
        msg = f"Need at least {min_folds} folds for PBO, got {n}"
        raise ValueError(msg)

    is_arr = np.array(is_sharpes)
    oos_arr = np.array(oos_sharpes)

    # IS Sharpe의 상대적 랭크 계산 (높을수록 높은 랭크)
    is_ranks = _rank_array(is_arr)
    oos_ranks = _rank_array(oos_arr)

    # IS에서 best인 fold가 OOS에서 중위수 이하인 비율
    # logit 기반 PBO: IS에서 최고인 fold의 OOS 랭크 → logit 변환
    # 간소화: rank correlation 기반
    # PBO = IS에서 상위 절반인 fold들 중 OOS에서 하위 절반에 속하는 비율
    median_is_rank = np.median(is_ranks)
    median_oos_rank = np.median(oos_ranks)

    # IS에서 상위(높은 랭크)인 fold
    is_top_mask = is_ranks >= median_is_rank
    # 이 fold들 중 OOS에서 하위(낮은 랭크)인 비율
    is_top_count = int(is_top_mask.sum())
    if is_top_count == 0:
        return 0.5

    overfit_count = int(np.sum(is_top_mask & (oos_ranks < median_oos_rank)))
    return overfit_count / is_top_count


def calculate_pbo_logit(
    is_sharpes: list[float],
    oos_sharpes: list[float],
) -> float:
    """Logit 기반 PBO 계산 (Lopez de Prado 원본 방법).

    IS에서 최적 전략의 OOS 상대적 순위를 logit 변환하여
    과적합 확률을 추정합니다.

    Args:
        is_sharpes: CPCV fold별 IS Sharpe 목록
        oos_sharpes: CPCV fold별 OOS Sharpe 목록

    Returns:
        PBO (0~1). 0.5 미만이면 과적합 확률 낮음.
    """
    if len(is_sharpes) != len(oos_sharpes):
        msg = f"IS and OOS sharpes must have same length: {len(is_sharpes)} vs {len(oos_sharpes)}"
        raise ValueError(msg)

    n = len(is_sharpes)
    min_folds = 2
    if n < min_folds:
        msg = f"Need at least {min_folds} folds for PBO, got {n}"
        raise ValueError(msg)

    is_arr = np.array(is_sharpes)
    oos_arr = np.array(oos_sharpes)

    # IS에서 최적(최고 Sharpe) fold 찾기
    best_is_idx = int(np.argmax(is_arr))

    # 해당 fold의 OOS Sharpe 순위 (0~1 범위)
    oos_rank = _rank_array(oos_arr)
    relative_rank = oos_rank[best_is_idx] / n

    # 상대 순위가 0.5 이하이면 IS 최적이 OOS에서는 하위 → 과적합
    # logit 변환: logit(w) = log(w / (1 - w))
    # PBO = P(logit(w) <= 0) = P(w <= 0.5)
    # 간단히: 상대 순위 자체가 PBO의 역수
    return 1.0 - relative_rank


def _rank_array(arr: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    """배열의 순위 계산 (1-based, 높을수록 높은 랭크).

    Args:
        arr: 입력 배열

    Returns:
        순위 배열 (동률은 평균 순위)
    """
    from scipy.stats import rankdata

    return rankdata(arr, method="average")  # type: ignore[return-value]
