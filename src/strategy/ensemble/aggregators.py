"""Ensemble Aggregation Functions.

여러 서브 전략의 direction/strength를 결합하는 4가지 순수 함수.
모두 (combined_direction, combined_strength) 튜플을 반환합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops on DataFrame rows)
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight(
    directions: pd.DataFrame,
    strengths: pd.DataFrame,
    weights: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """동일 가중 평균 집계.

    각 전략의 strength를 가중 평균하여 앙상블 시그널을 생성합니다.

    Args:
        directions: (n_bars, n_strategies) direction 행렬 (-1/0/1)
        strengths: (n_bars, n_strategies) strength 행렬
        weights: (n_strategies,) 정적 가중치

    Returns:
        (combined_direction, combined_strength) 튜플
    """
    norm_weights = weights / weights.sum()

    # strength 가중 평균
    combined_strength = strengths.mul(norm_weights, axis=1).sum(axis=1)
    combined_direction = pd.Series(
        np.sign(combined_strength).fillna(0).astype(int),
        index=strengths.index,
    )

    return combined_direction, combined_strength


def inverse_volatility(
    directions: pd.DataFrame,
    strengths: pd.DataFrame,
    weights: pd.Series,
    vol_lookback: int = 63,
) -> tuple[pd.Series, pd.Series]:
    """변동성 역비례 가중 집계.

    strength의 rolling std가 낮은 (안정적인) 전략에 높은 가중치를 부여합니다.
    NaN 초기 구간은 equal_weight fallback.

    Args:
        directions: (n_bars, n_strategies) direction 행렬
        strengths: (n_bars, n_strategies) strength 행렬
        weights: (n_strategies,) 정적 가중치 (fallback용)
        vol_lookback: 변동성 계산 lookback 기간

    Returns:
        (combined_direction, combined_strength) 튜플
    """
    # 각 전략 strength의 rolling std
    rolling_vol = strengths.rolling(window=vol_lookback, min_periods=vol_lookback).std()

    # 역변동성 가중치: 1/vol (vol=0 방지)
    inv_vol = 1.0 / rolling_vol.clip(lower=1e-8)

    # 행별 정규화
    row_sum = inv_vol.sum(axis=1)
    inv_vol_weights = inv_vol.div(row_sum, axis=0)

    # NaN 구간 fallback (EW)
    ew_fallback_dir, ew_fallback_str = equal_weight(directions, strengths, weights)

    # 가중 합산
    combined_strength = (strengths * inv_vol_weights).sum(axis=1)
    combined_direction = pd.Series(
        np.sign(combined_strength).fillna(0).astype(int),
        index=strengths.index,
    )

    # NaN 마스크: rolling_vol 전체 NaN인 행
    nan_mask: pd.Series = rolling_vol.isna().all(axis=1)  # type: ignore[assignment]
    fill_mask = nan_mask.astype(bool)
    combined_direction = combined_direction.where(~fill_mask, ew_fallback_dir)
    combined_strength = combined_strength.where(~fill_mask, ew_fallback_str)

    return combined_direction, combined_strength


def majority_vote(
    directions: pd.DataFrame,
    strengths: pd.DataFrame,
    weights: pd.Series,
    min_agreement: float = 0.5,
) -> tuple[pd.Series, pd.Series]:
    """다수결 합의 집계.

    전략 방향의 합의율이 min_agreement 이상일 때만 시그널 생성.
    합의율 미달 시 중립(0).

    Args:
        directions: (n_bars, n_strategies) direction 행렬 (-1/0/1)
        strengths: (n_bars, n_strategies) strength 행렬
        weights: (n_strategies,) 정적 가중치 (미사용, 인터페이스 통일)
        min_agreement: 최소 합의 비율 (0.0~1.0)

    Returns:
        (combined_direction, combined_strength) 튜플
    """
    # 각 행에서 Long/Short 전략 수
    long_count = (directions > 0).sum(axis=1)
    short_count = (directions < 0).sum(axis=1)

    # 활성 전략 수 (direction != 0)
    active_count = (directions != 0).sum(axis=1).clip(lower=1)

    # 합의율
    long_agreement = long_count / active_count
    short_agreement = short_count / active_count

    # direction 결정
    combined_direction = pd.Series(0, index=directions.index, dtype=int)
    combined_direction = combined_direction.where(~(long_agreement >= min_agreement), 1)
    combined_direction = combined_direction.where(~(short_agreement >= min_agreement), -1)

    # strength: agreement_ratio * avg(|strength|)
    avg_abs_strength = strengths.abs().mean(axis=1)
    agreement_ratio = pd.Series(0.0, index=directions.index)
    agreement_ratio = agreement_ratio.where(
        combined_direction == 0,
        np.where(combined_direction > 0, long_agreement, short_agreement),
    )
    combined_strength = combined_direction * agreement_ratio * avg_abs_strength

    return combined_direction, combined_strength


def strategy_momentum(
    directions: pd.DataFrame,
    strengths: pd.DataFrame,
    weights: pd.Series,
    momentum_lookback: int = 126,
    top_n: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """최근 성과 기반 모멘텀 가중 집계.

    direction x |strength|를 return proxy로 사용하여 rolling Sharpe를 계산하고,
    상위 top_n 전략만 선택하여 EW 집계합니다.
    NaN 초기 구간은 equal_weight fallback.

    Args:
        directions: (n_bars, n_strategies) direction 행렬
        strengths: (n_bars, n_strategies) strength 행렬
        weights: (n_strategies,) 정적 가중치 (fallback용)
        momentum_lookback: 모멘텀 계산 lookback 기간
        top_n: 선택할 상위 전략 수

    Returns:
        (combined_direction, combined_strength) 튜플
    """
    # Return proxy: direction x |strength|
    proxy_returns = directions * strengths.abs()

    # Rolling Sharpe: mean / std (annualization 미적용, 상대 비교용)
    rolling_mean = proxy_returns.rolling(
        window=momentum_lookback, min_periods=momentum_lookback
    ).mean()
    rolling_std = proxy_returns.rolling(
        window=momentum_lookback, min_periods=momentum_lookback
    ).std()
    rolling_sharpe = rolling_mean / rolling_std.clip(lower=1e-8)

    # 상위 top_n 선택 마스크
    # rank: ascending=False → 1위=가장 높은 Sharpe
    ranks = rolling_sharpe.rank(axis=1, ascending=False, method="min")
    top_mask = ranks <= top_n

    # 선택된 전략만 strength 사용, 나머지 0
    selected_strengths = strengths.where(top_mask, 0.0)

    # 선택된 전략 수 (행별)
    n_selected = top_mask.sum(axis=1).clip(lower=1)

    # EW of selected
    combined_strength = selected_strengths.sum(axis=1) / n_selected
    combined_direction = pd.Series(
        np.sign(combined_strength).fillna(0).astype(int),
        index=strengths.index,
    )

    # NaN 구간 fallback (rolling_sharpe 전체 NaN)
    nan_mask = rolling_sharpe.isna().all(axis=1)
    ew_dir, ew_str = equal_weight(directions, strengths, weights)
    combined_direction = combined_direction.where(~nan_mask, ew_dir)
    combined_strength = combined_strength.where(~nan_mask, ew_str)

    return combined_direction, combined_strength
