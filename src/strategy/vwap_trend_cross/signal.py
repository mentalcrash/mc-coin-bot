"""VWAP Trend Crossover signal generator.

Long when short VWAP > long VWAP AND close > short VWAP.
Short when short VWAP < long VWAP AND close < short VWAP.
Conviction scaled by normalized VWAP spread magnitude.

Shift(1) Rule: all features shifted by 1 bar before signal computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals
from src.strategy.vwap_trend_cross.config import ShortMode

if TYPE_CHECKING:
    from src.strategy.vwap_trend_cross.config import VwapTrendCrossConfig


def generate_signals(df: pd.DataFrame, config: VwapTrendCrossConfig) -> StrategySignals:
    """Generate VWAP Trend Crossover signals.

    Args:
        df: Preprocessed DataFrame (output of preprocess())
        config: Strategy configuration

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    # --- Shift(1): use previous bar's indicators ---
    vwap_short = df["vwap_short"].shift(1)
    vwap_long = df["vwap_long"].shift(1)
    vwap_spread = df["vwap_spread"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)
    close_prev = df["close"].shift(1)

    # --- Signal Logic ---
    # Bullish: short VWAP > long VWAP AND price above short VWAP
    long_signal = (vwap_short > vwap_long) & (close_prev > vwap_short)
    # Bearish: short VWAP < long VWAP AND price below short VWAP
    short_signal = (vwap_short < vwap_long) & (close_prev < vwap_short)

    # --- Direction (ShortMode 3-way) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength: direction * vol_scalar * abs(vwap_spread) / spread_clip ---
    # Normalize conviction to [0, 1] using spread magnitude
    conviction = (vwap_spread.abs() / config.spread_clip).clip(upper=1.0).fillna(0)
    strength = direction.astype(float) * vol_scalar.fillna(0) * conviction

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
    config: VwapTrendCrossConfig,
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
