"""Volume MACD Momentum signal generator.

VMACD histogram > 0 (volume expansion) + price momentum agreement = entry.
VMACD as primary, price momentum as confirmation.

Shift(1) Rule: all features shifted by 1 bar before signal computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals
from src.strategy.vmacd_mom.config import ShortMode

if TYPE_CHECKING:
    from src.strategy.vmacd_mom.config import VmacdMomConfig


def generate_signals(df: pd.DataFrame, config: VmacdMomConfig) -> StrategySignals:
    """Generate Volume MACD Momentum signals.

    Args:
        df: Preprocessed DataFrame (output of preprocess())
        config: Strategy configuration

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    # --- Shift(1): use previous bar's indicators ---
    vmacd_hist = df["vmacd_hist"].shift(1)
    mom_return = df["mom_return"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    # VMACD > 0 = volume expansion (primary)
    vol_expanding = vmacd_hist > 0
    mom_dir = np.sign(mom_return)

    # Long: volume expanding + positive price momentum
    long_signal = vol_expanding & (mom_dir > 0)
    # Short: volume expanding + negative price momentum
    short_signal = vol_expanding & (mom_dir < 0)

    # --- Direction (ShortMode 3-way) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength: direction * vol_scalar ---
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
    config: VmacdMomConfig,
) -> pd.Series:
    """Compute direction with ShortMode 3-way branching."""
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
        raw = np.where(
            long_signal,
            1,
            np.where(short_signal, -1, 0),
        )

    return pd.Series(raw, index=df.index, dtype=int)
