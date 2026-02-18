"""Funding Pressure Trend 시그널 생성.

SMA cross로 추세 방향, FR z-score로 리스크 필터.
Shift(1) Rule: 모든 feature에 shift(1) 적용.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.fr_press_trend.config import ShortMode
from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.fr_press_trend.config import FrPressTrendConfig


def generate_signals(df: pd.DataFrame, config: FrPressTrendConfig) -> StrategySignals:
    """Funding Pressure Trend 시그널 생성.

    Signal Logic:
        1. trend_dir = sign(sma_fast - sma_slow)
        2. trend_active = ER > er_threshold
        3. FR aligned: long & fr_z < threshold, short & fr_z > -threshold
        4. FR extreme against: long & fr_z > extreme, short & fr_z < -extreme -> BLOCK
        5. FR NaN -> trend_dir * trend_active (trend-only fallback)

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    # --- Shift(1): 전봉 기준 ---
    sma_fast = df["sma_fast"].shift(1)
    sma_slow = df["sma_slow"].shift(1)
    er = df["er"].shift(1)
    fr_z = df["fr_z"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Trend Direction ---
    trend_dir = np.sign(sma_fast - sma_slow)
    trend_active = er > config.er_threshold

    # --- FR Filter ---
    fr_valid = fr_z.notna()

    # FR aligned: 방향과 FR이 일치 (과열되지 않음)
    fr_aligned_long = fr_z < config.fr_aligned_threshold
    fr_aligned_short = fr_z > -config.fr_aligned_threshold
    fr_aligned = pd.Series(
        np.where(trend_dir > 0, fr_aligned_long, np.where(trend_dir < 0, fr_aligned_short, True)),
        index=df.index,
    )

    # FR extreme against: 방향 반대 극단 FR -> BLOCK
    fr_extreme_long = fr_z > config.fr_extreme_threshold
    fr_extreme_short = fr_z < -config.fr_extreme_threshold
    fr_extreme_against = pd.Series(
        np.where(trend_dir > 0, fr_extreme_long, np.where(trend_dir < 0, fr_extreme_short, False)),
        index=df.index,
    )

    # --- Direction ---
    # Case 1: FR valid & aligned & not extreme -> trend_dir
    # Case 2: FR valid & extreme against -> 0 (blocked)
    # Case 3: FR invalid (NaN) -> trend_dir (fallback, trend-only)
    direction_raw = pd.Series(
        np.where(
            fr_valid,
            np.where(fr_extreme_against, 0, np.where(fr_aligned, trend_dir, trend_dir)),
            trend_dir,  # FR NaN fallback
        ),
        index=df.index,
    )

    # Apply trend quality gate (fillna(0) for warmup NaN)
    gated_dir = pd.Series(
        np.where(trend_active, direction_raw.fillna(0), 0),
        index=df.index,
        dtype=int,
    )

    long_signal = gated_dir > 0
    short_signal = gated_dir < 0

    # --- Direction (ShortMode 3-way) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength ---
    strength = direction.astype(float) * vol_scalar.fillna(0)

    if config.short_mode == ShortMode.HEDGE_ONLY:
        strength = pd.Series(
            np.where(direction == -1, strength * config.hedge_strength_ratio, strength),
            index=df.index,
        )

    strength = strength.fillna(0.0)

    # --- Entries / Exits ---
    prev_dir = direction.shift(1).fillna(0).astype(int)
    entries = (direction != 0) & (direction != prev_dir)
    exits = (direction == 0) & (prev_dir != 0)

    return StrategySignals(
        entries=entries.astype(bool),
        exits=exits.astype(bool),
        direction=direction,
        strength=strength,
    )


def _compute_direction(
    long_signal: pd.Series,
    short_signal: pd.Series,
    df: pd.DataFrame,
    config: FrPressTrendConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    if config.short_mode == ShortMode.DISABLED:
        raw = np.where(long_signal, 1, 0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        dd = df["drawdown"].shift(1)
        hedge_active = dd < config.hedge_threshold
        raw = np.where(
            long_signal,
            1,
            np.where(short_signal & hedge_active, -1, 0),
        )

    else:  # FULL
        raw = np.where(long_signal, 1, np.where(short_signal, -1, 0))

    return pd.Series(raw, index=df.index, dtype=int)
