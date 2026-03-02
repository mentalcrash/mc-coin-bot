"""Carry-Regime Trend 시그널 생성.

Multi-scale EMA alignment으로 trend entry,
FR percentile로 exit speed 적응 조절.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.carry_regime_12h.config import CarryRegimeConfig


def generate_signals(df: pd.DataFrame, config: CarryRegimeConfig) -> StrategySignals:
    """Carry-Regime Trend 시그널 생성.

    Entry: EMA alignment score > threshold → long/short
    Exit: EMA alignment 약화 시 exit. FR percentile이 높으면 exit threshold 하향
          (extreme carry = crash risk → 빠른 exit).

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.carry_regime_12h.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    alignment = df["ema_alignment"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)
    fr_pct = df["fr_percentile"].shift(1).fillna(0.5)

    # --- Adaptive Exit Threshold ---
    # Higher FR percentile → lower exit threshold → faster exit
    # exit_threshold = exit_base - carry_sensitivity * (fr_percentile - 0.5)
    # At fr_pct=0.5 (median): threshold = exit_base
    # At fr_pct=1.0 (extreme): threshold = exit_base - 0.5 * sensitivity
    # At fr_pct=0.0 (calm): threshold = exit_base + 0.5 * sensitivity
    exit_threshold: pd.Series = (  # type: ignore[assignment]
        config.exit_base_threshold - config.carry_sensitivity * (fr_pct - 0.5)
    )
    exit_threshold = exit_threshold.clip(lower=0.0, upper=1.0)

    # --- Entry/Exit Conditions ---
    # Entry threshold is fixed (full alignment needed for entry)
    entry_threshold = 0.6  # alignment >= 0.6 → at least 2/3 EMA aligned

    long_entry = alignment >= entry_threshold
    short_entry = alignment <= -entry_threshold

    # Exit: alignment drops below adaptive threshold
    long_exit = alignment < exit_threshold
    short_exit = alignment > -exit_threshold

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_entry=long_entry,
        short_entry=short_entry,
        long_exit=long_exit,
        short_exit=short_exit,
        df=df,
        config=config,
    )

    # --- Strength ---
    # Conviction = |alignment| scaled by vol_scalar
    conviction = alignment.abs()
    strength = direction.astype(float) * vol_scalar.fillna(0) * conviction.fillna(0)

    if config.short_mode == ShortMode.HEDGE_ONLY:
        strength = pd.Series(
            np.where(
                direction == -1,
                strength * config.hedge_strength_ratio,
                strength,
            ),
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
    long_entry: pd.Series,
    short_entry: pd.Series,
    long_exit: pd.Series,
    short_exit: pd.Series,
    df: pd.DataFrame,
    config: CarryRegimeConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산.

    Trend-following state machine:
    - Entry: alignment crosses entry_threshold
    - Exit: alignment drops below adaptive exit_threshold
    """
    from src.strategy.carry_regime_12h.config import ShortMode

    if config.short_mode == ShortMode.DISABLED:
        # Long only: enter on alignment up, exit when alignment drops
        raw = np.where(long_entry & ~long_exit, 1, 0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        dd = df["drawdown"].shift(1)
        hedge_active = dd < config.hedge_threshold
        raw = np.where(
            long_entry & ~long_exit,
            1,
            np.where(short_entry & ~short_exit & hedge_active, -1, 0),
        )

    else:  # FULL
        raw = np.where(
            long_entry & ~long_exit,
            1,
            np.where(short_entry & ~short_exit, -1, 0),
        )

    return pd.Series(raw, index=df.index, dtype=int)
