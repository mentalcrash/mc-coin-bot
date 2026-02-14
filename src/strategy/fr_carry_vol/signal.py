"""Funding Rate Carry (Vol-Conditioned) 시그널 생성.

극단 FR → contrarian carry, 저변동성 환경에서만 활성화.
Shift(1) Rule: 모든 feature에 shift(1) 적용.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.fr_carry_vol.config import ShortMode
from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.fr_carry_vol.config import FRCarryVolConfig


def generate_signals(df: pd.DataFrame, config: FRCarryVolConfig) -> StrategySignals:
    """FR Carry Vol 시그널 생성.

    Signal Logic:
        1. |funding_zscore| > fr_extreme_zscore → carry 기회
        2. vol_pctile < vol_condition_pctile → 저변동성 OK
        3. direction = -sign(avg_FR) (positive FR → short, negative → long)

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    # --- Shift(1): 전봉 기준 ---
    avg_fr = df["avg_funding_rate"].shift(1)
    fz = df["funding_zscore"].shift(1)
    vol_pctile = df["vol_pctile"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    # 1. Extreme FR z-score
    extreme_fr = fz.abs() > config.fr_extreme_zscore

    # 2. FR above entry threshold
    above_threshold = avg_fr.abs() > config.fr_entry_threshold

    # 3. Low-vol conditioning
    low_vol = vol_pctile < config.vol_condition_pctile

    # 4. Carry direction: -sign(avg_FR)
    carry_dir = -np.sign(avg_fr)

    # Combined: entry when extreme FR + above threshold + low vol
    carry_active = extreme_fr & above_threshold & low_vol

    long_signal = carry_active & (carry_dir > 0)
    short_signal = carry_active & (carry_dir < 0)

    # --- Direction (ShortMode 3-way) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength ---
    # Conviction from z-score magnitude (clamped 1~3)
    conviction = fz.abs().clip(upper=3.0).fillna(0) / 3.0
    strength = direction.astype(float) * vol_scalar.fillna(0) * conviction.fillna(0)

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
    config: FRCarryVolConfig,
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
