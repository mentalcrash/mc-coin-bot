"""Adaptive FR Carry 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
FR 극단 이벤트 기반 → carry-forward position until exit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.adaptive_fr_carry.config import AdaptiveFrCarryConfig


def generate_signals(df: pd.DataFrame, config: AdaptiveFrCarryConfig) -> StrategySignals:
    """Adaptive FR Carry 시그널 생성.

    Entry: |fr_z| > entry_threshold AND atr_ratio < vol_ratio_exit AND er < er_max
    Direction: fr_z > 0 → SHORT (양 FR carry 수취), fr_z < 0 → LONG (음 FR carry 수취)
    Exit: |fr_z| < exit_threshold OR atr_ratio > vol_ratio_exit OR hold_bars > max_hold

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    # --- Shift(1): 전봉 기준 시그널 ---
    fr_z = df["fr_z"].shift(1)
    atr_ratio = df["atr_ratio"].shift(1)
    er = df["er"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Entry / Exit Conditions ---
    vol_safe = atr_ratio < config.vol_ratio_exit
    no_trend = er < config.er_max

    # Entry: FR 극단 + 변동성 안전 + 추세 없음
    long_signal = (fr_z < -config.fr_entry_threshold) & vol_safe & no_trend
    short_signal = (fr_z > config.fr_entry_threshold) & vol_safe & no_trend

    # Exit: FR 정상 복귀 OR 변동성 폭발
    exit_by_fr = fr_z.abs() < config.fr_exit_threshold
    exit_by_vol = atr_ratio > config.vol_ratio_exit

    # --- State Machine: carry-forward with max hold ---
    direction = _carry_forward_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        exit_by_fr=exit_by_fr,
        exit_by_vol=exit_by_vol,
        max_hold_bars=config.max_hold_bars,
        short_mode=config.short_mode,
        index=df.index,
    )

    # --- Strength ---
    fr_strength = (fr_z.abs() / 3.0).clip(upper=1.0)
    base_scalar = vol_scalar.fillna(0) * fr_strength.fillna(0)
    strength = direction.astype(float) * base_scalar
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


def _carry_forward_direction(
    long_signal: pd.Series,
    short_signal: pd.Series,
    exit_by_fr: pd.Series,
    exit_by_vol: pd.Series,
    max_hold_bars: int,
    short_mode: int,
    index: pd.Index,
) -> pd.Series:
    """FR carry state machine with max hold limit."""
    from src.strategy.adaptive_fr_carry.config import ShortMode

    n = len(index)
    long_arr = long_signal.to_numpy(dtype=bool, na_value=False)
    short_arr = short_signal.to_numpy(dtype=bool, na_value=False)
    exit_fr = exit_by_fr.to_numpy(dtype=bool, na_value=False)
    exit_vol = exit_by_vol.to_numpy(dtype=bool, na_value=False)

    direction = np.zeros(n, dtype=int)
    pos = 0
    hold_count = 0

    for i in range(n):
        if pos != 0:
            hold_count += 1
            # Exit conditions
            if exit_fr[i] or exit_vol[i] or hold_count > max_hold_bars:
                pos = 0
                hold_count = 0

        if pos == 0:
            if long_arr[i]:
                pos = 1
                hold_count = 1
            elif short_arr[i]:
                pos = 0 if short_mode == ShortMode.DISABLED else -1
                hold_count = 1 if pos != 0 else 0

        direction[i] = pos

    return pd.Series(direction, index=index, dtype=int)
