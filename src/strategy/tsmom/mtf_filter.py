"""Multi-Timeframe Filter for TSMOM Strategy.

상위 타임프레임 추세를 참조하여 하위 타임프레임 시그널을 필터링합니다.

Design Principles:
    - 독립 모듈: 기존 signal.py 수정 없이 후처리로 적용
    - 벡터화: Pandas 연산만 사용 (루프 금지)
    - Shift(1) Rule: 상위 TF 시그널도 전봉 기준

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns
    - Shift(1) Rule: 미래 참조 편향 방지

Example:
    >>> from src.strategy.tsmom.mtf_filter import (
    ...     compute_htf_trend,
    ...     align_htf_to_ltf,
    ...     apply_mtf_filter,
    ... )
    >>> htf_trend = compute_htf_trend(weekly_df, lookback=4)
    >>> htf_aligned = align_htf_to_ltf(htf_trend, daily_df.index)
    >>> filtered_signals = apply_mtf_filter(signals, htf_aligned, config)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.tsmom.config import MTFFilterConfig, MTFFilterMode
from src.strategy.types import Direction, StrategySignals

__all__ = [
    "align_htf_to_ltf",
    "apply_mtf_filter",
    "compute_htf_trend",
]


def compute_htf_trend(
    htf_df: pd.DataFrame,
    lookback: int,
    use_log_returns: bool = True,
) -> pd.Series:
    """상위 타임프레임 추세 방향 계산.

    누적 수익률 기반 모멘텀을 계산하고 방향(-1, 0, 1)을 반환합니다.
    Shift(1) Rule이 적용되어 현재 봉에서 전봉까지의 데이터만 사용합니다.

    Args:
        htf_df: 상위 TF OHLCV DataFrame (DatetimeIndex 필수)
        lookback: 모멘텀 계산 기간 (캔들 수)
        use_log_returns: 로그 수익률 사용 여부 (권장: True)

    Returns:
        추세 방향 Series (-1=하락, 0=중립, 1=상승)

    Example:
        >>> htf_trend = compute_htf_trend(weekly_df, lookback=4)
        >>> htf_trend.tail()
        2025-01-27    1  # 상승 추세
        2025-02-03   -1  # 하락 추세
    """
    close = pd.Series(htf_df["close"])

    # 수익률 계산
    if use_log_returns:
        returns: pd.Series = pd.Series(np.log(close / close.shift(1)), index=close.index)
    else:
        returns = close.pct_change()

    # 누적 수익률 기반 모멘텀 (lookback 기간)
    rolling_sum = returns.rolling(window=lookback, min_periods=lookback).sum()
    momentum = pd.Series(rolling_sum, index=returns.index)

    # Shift(1): 현재 봉 기준으로 전봉까지의 모멘텀 사용
    shifted = momentum.shift(1)
    momentum_shifted = pd.Series(shifted, index=momentum.index)

    # 방향 추출 (-1, 0, 1)
    # np.sign은 0을 0으로 반환 (NaN은 NaN 유지)
    trend = pd.Series(
        np.sign(momentum_shifted),
        index=htf_df.index,
        name="htf_trend",
    )

    logger.debug(
        f"HTF Trend | periods: {len(htf_df)}, lookback: {lookback}, bullish: {(trend == 1).sum()}, bearish: {(trend == -1).sum()}"
    )

    return trend


def align_htf_to_ltf(
    htf_trend: pd.Series,
    ltf_index: pd.Index,
) -> pd.Series:
    """상위 TF 추세를 하위 TF 인덱스에 정렬 (Forward-Fill).

    Weekly 데이터를 Daily 인덱스에 맞춰 정렬합니다.
    각 Daily 봉은 해당 시점까지 확정된 Weekly 추세를 참조합니다.

    Args:
        htf_trend: 상위 TF 추세 Series (예: Weekly DatetimeIndex)
        ltf_index: 하위 TF DatetimeIndex (예: Daily)

    Returns:
        하위 TF 인덱스에 정렬된 추세 Series (-1, 0, 1)

    Example:
        >>> # Weekly: [2025-01-27: 1, 2025-02-03: -1]
        >>> # Daily: 2025-01-27, 01-28, 01-29, 01-30, 01-31, 02-03, 02-04
        >>> aligned = align_htf_to_ltf(htf_trend, daily_index)
        >>> # Result: [1, 1, 1, 1, 1, -1, -1]
    """
    # 상위 TF 인덱스를 하위 TF 인덱스로 재인덱싱 (Forward-Fill)
    aligned = htf_trend.reindex(ltf_index, method="ffill")

    # 초기 NaN은 0 (중립)으로 처리 - 상위 TF 데이터가 없는 초기 구간
    aligned = aligned.fillna(0).astype(int)

    logger.debug(
        f"HTF Alignment | HTF: {len(htf_trend)}, LTF: {len(ltf_index)}, bullish: {(aligned == 1).sum()}, bearish: {(aligned == -1).sum()}"
    )

    return aligned


def apply_mtf_filter(
    signals: StrategySignals,
    htf_aligned: pd.Series,
    config: MTFFilterConfig,
) -> StrategySignals:
    """MTF 필터를 적용하여 시그널 수정.

    Args:
        signals: 원본 시그널 (generate_signals 출력)
        htf_aligned: 정렬된 상위 TF 추세 (-1, 0, 1)
        config: MTF 필터 설정

    Returns:
        필터링된 StrategySignals

    Modes:
        - CONSENSUS: htf_trend == ltf_direction 일 때만 시그널 유지
        - VETO: htf_trend와 반대일 때만 필터링 (중립은 통과)
        - WEIGHTED: 방향에 따라 strength 조절
    """
    direction = signals.direction.copy()
    strength = signals.strength.copy()
    entries = signals.entries.copy()

    # Direction enum 값으로 비교 (int로 변환)
    dir_long = int(Direction.LONG)
    dir_short = int(Direction.SHORT)
    dir_neutral = int(Direction.NEUTRAL)

    if config.mode == MTFFilterMode.CONSENSUS:
        # 상위 TF와 동일 방향일 때만 시그널 유지
        # 또는 현재 시그널이 중립이면 통과
        aligned_mask = (direction == htf_aligned) | (direction == dir_neutral)

        # 불일치 시 중립으로 변경
        direction = direction.where(aligned_mask, dir_neutral)
        strength = strength.where(aligned_mask, 0.0)
        entries = entries & aligned_mask

        filtered_count = int((~aligned_mask & (signals.direction != dir_neutral)).sum())
        logger.info(f"MTF CONSENSUS | Filtered: {filtered_count} signals")

    elif config.mode == MTFFilterMode.VETO:
        # 상위 TF가 반대 방향일 때만 필터링 (중립은 통과)
        opposite_mask = ((direction == dir_long) & (htf_aligned == dir_short)) | (
            (direction == dir_short) & (htf_aligned == dir_long)
        )

        direction = direction.where(~opposite_mask, dir_neutral)
        strength = strength.where(~opposite_mask, 0.0)
        entries = entries & ~opposite_mask

        filtered_count = int(opposite_mask.sum())
        logger.info(f"MTF VETO | Vetoed: {filtered_count} opposite signals")

    elif config.mode == MTFFilterMode.WEIGHTED:
        # 방향에 따라 가중치 적용
        aligned_mask = direction == htf_aligned
        neutral_htf_mask = htf_aligned == 0
        against_mask = ((direction == dir_long) & (htf_aligned == dir_short)) | (
            (direction == dir_short) & (htf_aligned == dir_long)
        )

        # 가중치 배열 생성 (기본 1.0)
        weights = pd.Series(1.0, index=strength.index)
        weights = weights.mask(aligned_mask, config.weight_aligned)
        weights = weights.mask(neutral_htf_mask, config.weight_neutral)
        weights = weights.mask(against_mask, config.weight_against)

        # 강도에 가중치 적용
        strength = strength * weights

        # weight_against=0이면 해당 시그널 제거
        if config.weight_against == 0:
            direction = direction.where(~against_mask, dir_neutral)
            entries = entries & ~against_mask

        logger.info(
            f"MTF WEIGHTED | Aligned: {int(aligned_mask.sum())}, Neutral HTF: {int(neutral_htf_mask.sum())}, Against: {int(against_mask.sum())}"
        )

    # 청산 시그널 재계산 (direction 변경으로 인해)
    prev_direction = direction.shift(1).fillna(0)
    to_neutral = (direction == dir_neutral) & (prev_direction != dir_neutral)
    reversal = direction * prev_direction < 0
    exits = to_neutral | reversal

    return StrategySignals(
        entries=entries,
        exits=exits,
        direction=direction,
        strength=strength,
    )
