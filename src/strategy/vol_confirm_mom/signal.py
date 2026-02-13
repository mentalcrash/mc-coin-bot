"""Volume-Confirmed Momentum signal generator.

Enter momentum trade only when volume trend confirms: vol_sma_short > vol_sma_long.
Conviction scaled by volume ratio.

Shift(1) Rule: all features shifted by 1 bar before signal computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals
from src.strategy.vol_confirm_mom.config import ShortMode

if TYPE_CHECKING:
    from src.strategy.vol_confirm_mom.config import VolConfirmMomConfig


def generate_signals(df: pd.DataFrame, config: VolConfirmMomConfig) -> StrategySignals:
    """Generate Volume-Confirmed Momentum signals.

    Args:
        df: Preprocessed DataFrame (output of preprocess())
        config: Strategy configuration

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    # --- Shift(1): use previous bar's indicators ---
    mom_return = df["mom_return"].shift(1)
    vol_rising = df["vol_rising"].shift(1).fillna(value=False).astype(bool)
    vol_ratio = df["vol_ratio"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    mom_dir = np.sign(mom_return)

    # Long: positive momentum + rising volume
    long_signal = pd.Series((mom_dir > 0) & vol_rising, index=df.index)
    # Short: negative momentum + rising volume
    short_signal = pd.Series((mom_dir < 0) & vol_rising, index=df.index)

    # --- Direction (ShortMode 3-way) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength: direction * vol_scalar * vol_ratio (conviction) ---
    # vol_ratio > 1 means short vol > long vol = stronger conviction
    conviction = (vol_ratio.fillna(1.0) - 1.0).clip(lower=0.0) + 1.0
    # Normalize conviction to ~0.5~1.0 range
    norm_conviction = conviction / config.vol_ratio_clip
    strength = direction.astype(float) * vol_scalar.fillna(0) * norm_conviction.fillna(0)

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
    config: VolConfirmMomConfig,
) -> pd.Series:
    """Compute direction with ShortMode 3-way branching."""
    if config.short_mode == ShortMode.DISABLED:
        raw = np.where(long_signal, 1, 0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        drawdown = df["drawdown"].shift(1)
        hedge_active = drawdown < config.hedge_threshold
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
