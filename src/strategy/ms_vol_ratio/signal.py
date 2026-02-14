"""Multi-Scale Volatility Ratio 시그널 생성.

short_vol/long_vol ratio > upper → breakout (follow momentum).
Shift(1) Rule: 모든 feature에 shift(1) 적용.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.ms_vol_ratio.config import ShortMode
from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.ms_vol_ratio.config import MSVolRatioConfig


def generate_signals(df: pd.DataFrame, config: MSVolRatioConfig) -> StrategySignals:
    """Multi-Scale Volatility Ratio 시그널 생성.

    Signal Logic:
        1. vol_ratio > ratio_upper → information arrival (breakout)
        2. vol_ratio < ratio_lower → calm (no trade)
        3. Direction from momentum: sign(mom_return)

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    # --- Shift(1): 전봉 기준 ---
    vol_ratio = df["vol_ratio"].shift(1)
    mom_return = df["mom_return"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    # Breakout regime: short vol >> long vol
    breakout_regime = vol_ratio > config.ratio_upper

    # Momentum direction
    mom_dir = np.sign(mom_return)
    long_signal = breakout_regime & (mom_dir > 0)
    short_signal = breakout_regime & (mom_dir < 0)

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
    config: MSVolRatioConfig,
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
