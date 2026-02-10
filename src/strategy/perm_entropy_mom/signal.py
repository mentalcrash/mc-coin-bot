"""Permutation Entropy Momentum Signal Generator.

Low PE = orderly pattern = high conviction momentum.
High PE = noise = reduced/zero position.

Signal Formula:
    1. Shift(1) on pe_short, mom_direction, vol_scalar, conviction
    2. direction = sign(mom_direction) where PE < noise_threshold, else 0
    3. strength = direction * vol_scalar * conviction
    4. Apply short mode filtering

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - Shift(1) Rule: Lookahead bias prevention
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.perm_entropy_mom.config import PermEntropyMomConfig, ShortMode
from src.strategy.types import Direction, StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: PermEntropyMomConfig | None = None,
) -> StrategySignals:
    """Generate Permutation Entropy Momentum signals.

    Args:
        df: Preprocessed DataFrame (output of preprocess())
        config: PermEntropyMom configuration. Defaults to default config.

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: If required columns are missing
    """
    if config is None:
        config = PermEntropyMomConfig()

    required_cols = {"pe_short", "mom_direction", "vol_scalar", "conviction"}
    if config.short_mode == ShortMode.HEDGE_ONLY:
        required_cols.add("drawdown")

    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 1. Shift(1): use previous bar's indicators
    pe_short_prev: pd.Series = df["pe_short"].shift(1)  # type: ignore[assignment]
    mom_dir_prev: pd.Series = df["mom_direction"].shift(1)  # type: ignore[assignment]
    vol_scalar_prev: pd.Series = df["vol_scalar"].shift(1)  # type: ignore[assignment]
    conviction_prev: pd.Series = df["conviction"].shift(1)  # type: ignore[assignment]

    # 2. Noise gate: PE > noise_threshold -> neutral (pure noise)
    noise_mask = pe_short_prev > config.noise_threshold

    # 3. Direction = momentum direction where not noise, else 0
    direction_raw = pd.Series(
        np.where(noise_mask, 0, mom_dir_prev),
        index=df.index,
    )

    # 4. Strength = direction * vol_scalar * conviction
    strength_raw = direction_raw * vol_scalar_prev * conviction_prev

    # 5. Direction normalization
    direction = pd.Series(
        np.sign(strength_raw).fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    # 6. Strength (with NaN -> 0)
    strength = pd.Series(
        strength_raw.fillna(0),
        index=df.index,
        name="strength",
    )

    # 7. Short mode filtering
    if config.short_mode == ShortMode.DISABLED:
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        drawdown_series: pd.Series = df["drawdown"].shift(1)  # type: ignore[assignment]
        hedge_active = drawdown_series < config.hedge_threshold

        short_mask = direction == Direction.SHORT
        suppress_short = short_mask & ~hedge_active
        direction = direction.where(~suppress_short, Direction.NEUTRAL)
        strength = strength.where(~suppress_short, 0.0)

        active_short = short_mask & hedge_active
        strength = strength.where(
            ~active_short,
            strength * config.hedge_strength_ratio,
        )

        hedge_days = int(hedge_active.sum())
        if hedge_days > 0:
            logger.info(
                "Hedge Mode | Active: {} bars ({:.1f}%), Threshold: {:.1f}%",
                hedge_days,
                hedge_days / len(hedge_active) * 100,
                config.hedge_threshold * 100,
            )

    # 8. Entry/exit signals
    prev_direction = direction.shift(1).fillna(0)

    long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)
    short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)

    entries = pd.Series(
        long_entry | short_entry,
        index=df.index,
        name="entries",
    )

    to_neutral = (direction == Direction.NEUTRAL) & (prev_direction != Direction.NEUTRAL)
    reversal = direction * prev_direction < 0

    exits = pd.Series(
        to_neutral | reversal,
        index=df.index,
        name="exits",
    )

    # Debug: signal statistics
    valid_strength = strength[strength != 0]
    long_signals = strength[strength > 0]
    short_signals = strength[strength < 0]

    if len(valid_strength) > 0:
        logger.info(
            "Signal Statistics | Total: {} signals, Long: {} ({:.1f}%), Short: {} ({:.1f}%)",
            len(valid_strength),
            len(long_signals),
            len(long_signals) / len(valid_strength) * 100,
            len(short_signals),
            len(short_signals) / len(valid_strength) * 100,
        )

    return StrategySignals(
        entries=entries,
        exits=exits,
        direction=direction,
        strength=strength,
    )
