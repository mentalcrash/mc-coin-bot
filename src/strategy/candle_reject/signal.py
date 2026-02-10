"""Candlestick Rejection Momentum Signal Generator.

Rejection wick + volume confirmation -> directional momentum signal.

Signal Formula:
    1. Shift(1) on bull_reject, bear_reject, volume_zscore, body_position, vol_scalar
    2. bull_signal = (bull_reject_prev > threshold) AND (vol_zscore_prev > vol_threshold)
    3. bear_signal = (bear_reject_prev > threshold) AND (vol_zscore_prev > vol_threshold)
    4. Consecutive count: same-direction rejections -> boost
    5. strength = direction * vol_scalar * consecutive_boost
    6. Exit: body_position reversal OR timeout

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.candle_reject.config import CandleRejectConfig, ShortMode
from src.strategy.types import Direction, StrategySignals


def _count_consecutive(series: pd.Series) -> pd.Series:
    """같은 방향의 연속 횟수를 벡터화로 계산.

    Args:
        series: direction 시리즈 (-1, 0, 1)

    Returns:
        연속 횟수 시리즈 (방향 변경 시 리셋)
    """
    # 방향이 바뀌면 새 그룹
    group_id = (series != series.shift(1)).cumsum()
    consecutive: pd.Series = series.groupby(group_id).cumcount() + 1  # type: ignore[assignment]
    # 방향이 0인 경우 연속 횟수도 0
    consecutive = consecutive.where(series != 0, 0)
    return pd.Series(consecutive, index=series.index)


def _apply_exit_timeout(
    direction: pd.Series,
    timeout_bars: int,
) -> pd.Series:
    """타임아웃 기반 청산을 벡터화로 적용.

    진입 후 timeout_bars 이내에 방향이 유지되면 neutral로 전환합니다.

    Args:
        direction: 방향 시리즈 (-1, 0, 1)
        timeout_bars: 타임아웃 bar 수

    Returns:
        타임아웃 적용된 방향 시리즈
    """
    # 방향 변경 그룹 식별
    group_id = (direction != direction.shift(1)).cumsum()
    bars_in_position: pd.Series = direction.groupby(group_id).cumcount() + 1  # type: ignore[assignment]

    # 포지션 유지 기간이 timeout 초과 시 neutral
    timeout_mask = (direction != 0) & (bars_in_position > timeout_bars)
    result = direction.where(~timeout_mask, 0)
    return pd.Series(result, index=direction.index)


def generate_signals(
    df: pd.DataFrame,
    config: CandleRejectConfig | None = None,
) -> StrategySignals:
    """Candlestick Rejection Momentum 시그널 생성.

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
        config: Candlestick Rejection 설정. None이면 기본 설정 사용.

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    if config is None:
        config = CandleRejectConfig()

    required_cols = {"bull_reject", "bear_reject", "volume_zscore", "body_position", "vol_scalar"}
    if config.short_mode == ShortMode.HEDGE_ONLY:
        required_cols.add("drawdown")

    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # =========================================================================
    # 1. Shift(1) 적용: 전봉 기준 시그널
    # =========================================================================
    bull_reject_prev: pd.Series = df["bull_reject"].shift(1)  # type: ignore[assignment]
    bear_reject_prev: pd.Series = df["bear_reject"].shift(1)  # type: ignore[assignment]
    vol_zscore_prev: pd.Series = df["volume_zscore"].shift(1)  # type: ignore[assignment]
    vol_scalar_prev: pd.Series = df["vol_scalar"].shift(1)  # type: ignore[assignment]

    # =========================================================================
    # 2. Rejection + Volume confirmation
    # =========================================================================
    bull_signal = (bull_reject_prev > config.rejection_threshold) & (
        vol_zscore_prev > config.volume_zscore_threshold
    )
    bear_signal = (bear_reject_prev > config.rejection_threshold) & (
        vol_zscore_prev > config.volume_zscore_threshold
    )

    # =========================================================================
    # 3. Direction: 1 for bull rejection, -1 for bear rejection
    # =========================================================================
    direction_raw = pd.Series(
        np.where(bull_signal, 1, np.where(bear_signal, -1, 0)),
        index=df.index,
    )

    # =========================================================================
    # 4. Consecutive count & boost
    # =========================================================================
    consecutive_count = _count_consecutive(direction_raw)
    boost_mask = consecutive_count >= config.consecutive_min
    boost_factor = pd.Series(
        np.where(boost_mask, config.consecutive_boost, 1.0),
        index=df.index,
    )

    # =========================================================================
    # 5. Strength = direction * vol_scalar * boost
    # =========================================================================
    strength_raw = direction_raw * vol_scalar_prev * boost_factor

    # =========================================================================
    # 6. Exit timeout 적용
    # =========================================================================
    direction_with_timeout = _apply_exit_timeout(direction_raw, config.exit_timeout_bars)

    # Timeout으로 neutral이 된 bar의 strength도 0으로
    timeout_zeroed = direction_with_timeout == 0
    strength_raw = strength_raw.where(~(timeout_zeroed & (direction_raw != 0)), 0.0)

    # =========================================================================
    # 7. Direction 정규화
    # =========================================================================
    direction = pd.Series(
        np.sign(strength_raw).fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    # =========================================================================
    # 8. 강도 계산
    # =========================================================================
    strength = pd.Series(
        strength_raw.fillna(0),
        index=df.index,
        name="strength",
    )

    # =========================================================================
    # 9. 숏 모드에 따른 시그널 처리
    # =========================================================================
    if config.short_mode == ShortMode.DISABLED:
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        drawdown_series: pd.Series = df["drawdown"].shift(1)  # type: ignore[assignment]
        hedge_active = drawdown_series < config.hedge_threshold

        short_mask = direction == Direction.SHORT
        suppress_short = short_mask & ~hedge_active
        direction = direction.where(~suppress_short, Direction.NEUTRAL)
        strength = strength.where(~suppress_short, 0.0)

        active_short = short_mask & hedge_active
        strength = strength.where(
            ~active_short,
            strength * config.hedge_strength_ratio,
        )

        hedge_days = int(hedge_active.sum())
        if hedge_days > 0:
            logger.info(
                "Hedge Mode | Active: {} bars ({:.1f}%), Threshold: {:.1f}%",
                hedge_days,
                hedge_days / len(hedge_active) * 100,
                config.hedge_threshold * 100,
            )

    # =========================================================================
    # 10. 진입/청산 시그널
    # =========================================================================
    prev_direction = direction.shift(1).fillna(0)

    long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)
    short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)

    entries = pd.Series(
        long_entry | short_entry,
        index=df.index,
        name="entries",
    )

    # Exit: direction 변경 (neutral 전환 또는 반전) — timeout에 의한 neutral 포함
    to_neutral = (direction == Direction.NEUTRAL) & (prev_direction != Direction.NEUTRAL)
    reversal = direction * prev_direction < 0

    exits = pd.Series(
        to_neutral | reversal,
        index=df.index,
        name="exits",
    )

    # 디버그: 시그널 통계
    valid_strength = strength[strength != 0]
    long_signals = strength[strength > 0]
    short_signals = strength[strength < 0]

    if len(valid_strength) > 0:
        logger.info(
            "Signal Statistics | Total: {} signals, Long: {} ({:.1f}%), Short: {} ({:.1f}%)",
            len(valid_strength),
            len(long_signals),
            len(long_signals) / len(valid_strength) * 100,
            len(short_signals),
            len(short_signals) / len(valid_strength) * 100,
        )

    return StrategySignals(
        entries=entries,
        exits=exits,
        direction=direction,
        strength=strength,
    )
