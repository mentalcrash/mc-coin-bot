"""Asymmetric Volume Response 시그널 생성.

Impact asymmetry z-score > threshold → informed flow detected.
Direction from momentum + asymmetry sign.
Shift(1) Rule: 모든 feature에 shift(1) 적용.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.asym_vol_resp.config import ShortMode
from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.asym_vol_resp.config import AsymVolRespConfig


def generate_signals(df: pd.DataFrame, config: AsymVolRespConfig) -> StrategySignals:
    """Asymmetric Volume Response 시그널 생성.

    Signal Logic:
        1. |impact_asymmetry| > threshold → informed flow
        2. asymmetry > 0 + momentum > 0 → long (buying pressure)
        3. asymmetry < 0 + momentum < 0 → short (selling pressure)

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    # --- Shift(1): 전봉 기준 ---
    impact_asym = df["impact_asymmetry"].shift(1)
    mom_return = df["mom_return"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)
    volume_norm = df["volume_norm"].shift(1)

    # --- Signal Logic ---
    # Informed flow: high |asymmetry|
    informed = impact_asym.abs() > config.asym_threshold

    # High volume confirms
    high_volume = volume_norm > 0

    # Active signal: informed + high volume
    active = informed & high_volume

    # Direction: aligned asymmetry + momentum
    long_signal = active & (impact_asym > 0) & (mom_return > 0)
    short_signal = active & (impact_asym < 0) & (mom_return < 0)

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
    config: AsymVolRespConfig,
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
