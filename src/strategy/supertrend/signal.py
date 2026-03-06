"""SuperTrend 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
SuperTrend 초록(1) → Long, 빨강(-1) → Short(FULL) 또는 청산(DISABLED).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.supertrend.config import ShortMode
from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.supertrend.config import SuperTrendConfig


def generate_signals(df: pd.DataFrame, config: SuperTrendConfig) -> StrategySignals:
    """SuperTrend 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    # --- Shift(1): 전봉 기준 시그널 ---
    st_dir = df["supertrend_dir"].shift(1)

    # --- ADX 필터 ---
    adx_filter = pd.Series(True, index=df.index)
    if config.use_adx_filter:
        adx_val = df["adx"].shift(1)
        adx_filter = adx_val >= config.adx_threshold

    # --- Direction 계산 (ShortMode 반영) ---
    long_signal = (st_dir == 1) & adx_filter
    short_signal = (st_dir == -1) & adx_filter

    if config.short_mode == ShortMode.FULL:
        # Long/Short 양방향: +1, -1, 0
        direction = pd.Series(
            np.where(long_signal, 1, np.where(short_signal, -1, 0)),
            index=df.index,
            dtype=int,
        )
    else:
        # DISABLED (Long-only): +1, 0
        direction = pd.Series(
            np.where(long_signal, 1, 0),
            index=df.index,
            dtype=int,
        )

    # --- Strength 계산 ---
    if config.use_pyramiding and config.use_adx_filter:
        strength = _compute_pyramid_strength(df, direction, config)
    elif config.use_risk_sizing:
        prev_atr: pd.Series = df["atr"].shift(1)  # type: ignore[assignment]
        prev_close: pd.Series = df["close"].shift(1)  # type: ignore[assignment]
        stop_distance = (prev_atr * config.atr_stop_multiplier) / prev_close
        stop_distance = stop_distance.replace(0, np.nan)
        raw_strength = config.risk_per_trade / stop_distance
        strength = direction.astype(float) * raw_strength.fillna(0.0)
    else:
        strength = direction.astype(float)

    # --- Entries / Exits (상태 전환 감지) ---
    prev_dir = direction.shift(1).fillna(0).astype(int)
    entries = (direction != 0) & (prev_dir == 0)
    exits = (direction == 0) & (prev_dir != 0)

    return StrategySignals(
        entries=entries.astype(bool),
        exits=exits.astype(bool),
        direction=direction,
        strength=strength,
    )


def _compute_pyramid_strength(
    df: pd.DataFrame,
    direction: pd.Series,
    config: SuperTrendConfig,
) -> pd.Series:
    """분할 진입 strength 계산 (벡터화).

    Stage 1: ST 초록 + ADX >= threshold     → stage1_pct (40%)
    Stage 2: + close > Donchian High 돌파   → +stage2_pct (35%)
    Stage 3: + ADX >= adx_strong            → +stage3_pct (25%)

    Args:
        df: preprocess() 출력 DataFrame
        direction: direction 시리즈 (0 or 1)
        config: 전략 설정

    Returns:
        0.0 ~ 1.0 범위의 strength 시리즈
    """
    is_long = direction == 1

    # Stage 1: 기본 진입 (항상)
    strength = pd.Series(
        np.where(is_long, config.pyramid_stage1_pct, 0.0),
        index=df.index,
    )

    # Stage 2: 신고점 돌파 확인 (close > N봉 최고가, shift(1))
    prev_close: pd.Series = df["close"].shift(1)  # type: ignore[assignment]
    dc_high: pd.Series = (
        df["high"]
        .rolling(  # type: ignore[assignment]
            config.pyramid_high_period,
            min_periods=1,
        )
        .max()
        .shift(2)
    )  # shift(2): 전봉의 전 N봉 최고가
    new_high = prev_close > dc_high
    strength = strength + np.where(is_long & new_high, config.pyramid_stage2_pct, 0.0)

    # Stage 3: 추세 강화 (ADX >= strong threshold, shift(1))
    adx_val: pd.Series = df["adx"].shift(1)  # type: ignore[assignment]
    trend_strong = adx_val >= config.pyramid_adx_strong
    strength = strength + np.where(is_long & trend_strong, config.pyramid_stage3_pct, 0.0)

    # 최대 1.0으로 클램핑
    return strength.clip(upper=1.0)
