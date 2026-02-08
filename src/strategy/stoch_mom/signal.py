"""Stochastic Momentum Hybrid Signal Generator.

Generates entry/exit signals based on Stochastic %K/%D crossover
with SMA trend filter and ATR-based dynamic position sizing.

Signal Logic:
    - Long Entry: %K crosses above %D AND close > SMA
    - Short Entry: %K crosses below %D AND close < SMA (FULL mode)
    - Long Exit: %K crosses below %D (any crossover down)
    - Short Exit: %K crosses above %D (any crossover up)
    - strength = direction * vol_scalar * vol_ratio

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops, except state machine)
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: Lookahead bias prevention
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.stoch_mom.config import ShortMode
from src.strategy.types import Direction, StrategySignals

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.strategy.stoch_mom.config import StochMomConfig


def _compute_position_state(
    long_entry: NDArray[np.bool_],
    short_entry: NDArray[np.bool_],
    long_exit: NDArray[np.bool_],
    short_exit: NDArray[np.bool_],
    allow_short: bool,
) -> NDArray[np.int32]:
    """State machine based position tracking.

    Tracks position transitions:
        NEUTRAL -> LONG (on long_entry)
        NEUTRAL -> SHORT (on short_entry, if allow_short)
        LONG -> NEUTRAL (on long_exit)
        SHORT -> NEUTRAL (on short_exit)

    Args:
        long_entry: Long entry signal array
        short_entry: Short entry signal array
        long_exit: Long exit signal array (k_cross_down)
        short_exit: Short exit signal array (k_cross_up)
        allow_short: Whether short positions are allowed

    Returns:
        Position state array (-1, 0, 1)
    """
    n = len(long_entry)
    position = np.zeros(n, dtype=np.int32)

    for i in range(1, n):
        prev = position[i - 1]
        # Long entry
        if long_entry[i] and prev != Direction.LONG.value:
            position[i] = Direction.LONG.value
        # Short entry
        elif short_entry[i] and allow_short and prev != Direction.SHORT.value:
            position[i] = Direction.SHORT.value
        # Exit conditions
        elif (prev == Direction.LONG.value and long_exit[i]) or (
            prev == Direction.SHORT.value and short_exit[i]
        ):
            position[i] = Direction.NEUTRAL.value
        else:
            position[i] = prev

    return position


def generate_signals(
    df: pd.DataFrame,
    config: StochMomConfig,
) -> StrategySignals:
    """Generate Stochastic Momentum Hybrid signals.

    Signal Generation Pipeline:
        1. Stochastic %K/%D crossover detection (shift(1), shift(2))
        2. SMA trend filter (shift(1))
        3. State machine for position tracking
        4. ShortMode processing (DISABLED/FULL)
        5. Vol scalar * vol_ratio for strength
        6. Entry/Exit signal generation

    Args:
        df: Preprocessed DataFrame (preprocess() output)
            Required columns: pct_k, pct_d, sma, close, vol_scalar, vol_ratio
        config: Stochastic Momentum config

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: When required columns are missing
    """
    # Input validation
    required_cols = {"pct_k", "pct_d", "sma", "close", "vol_scalar", "vol_ratio"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # ================================================================
    # 1. Stochastic crossover detection (Shift(1) Rule)
    # ================================================================
    pct_k: pd.Series = df["pct_k"]  # type: ignore[assignment]
    pct_d: pd.Series = df["pct_d"]  # type: ignore[assignment]
    close: pd.Series = df["close"]  # type: ignore[assignment]
    sma: pd.Series = df["sma"]  # type: ignore[assignment]
    vol_scalar: pd.Series = df["vol_scalar"]  # type: ignore[assignment]
    vol_ratio: pd.Series = df["vol_ratio"]  # type: ignore[assignment]

    # Shift(1) and Shift(2): previous bars (lookahead prevention)
    prev_k: pd.Series = pct_k.shift(1)  # type: ignore[assignment]
    prev_d: pd.Series = pct_d.shift(1)  # type: ignore[assignment]
    prev_k2: pd.Series = pct_k.shift(2)  # type: ignore[assignment]
    prev_d2: pd.Series = pct_d.shift(2)  # type: ignore[assignment]

    # %K crosses above %D (bullish crossover)
    k_cross_up = (prev_k2 <= prev_d2) & (prev_k > prev_d)
    # %K crosses below %D (bearish crossover)
    k_cross_down = (prev_k2 >= prev_d2) & (prev_k < prev_d)

    # ================================================================
    # 2. Trend filter (Shift(1))
    # ================================================================
    prev_close: pd.Series = close.shift(1)  # type: ignore[assignment]
    prev_sma: pd.Series = sma.shift(1)  # type: ignore[assignment]

    trend_up = prev_close > prev_sma
    trend_down = prev_close < prev_sma

    # ================================================================
    # 3. Entry signals (crossover + trend filter)
    # ================================================================
    long_entry_raw = k_cross_up & trend_up
    short_entry_raw = k_cross_down & trend_down

    # Fill NaN with False (from shift)
    long_entry_raw = long_entry_raw.fillna(False)
    short_entry_raw = short_entry_raw.fillna(False)

    # Exit signals: opposite crossover (no trend filter needed)
    long_exit_raw = k_cross_down.fillna(False)
    short_exit_raw = k_cross_up.fillna(False)

    # ================================================================
    # 4. State machine for position tracking
    # ================================================================
    allow_short = config.short_mode == ShortMode.FULL
    position = _compute_position_state(
        long_entry_raw.to_numpy(),
        short_entry_raw.to_numpy(),
        long_exit_raw.to_numpy(),
        short_exit_raw.to_numpy(),
        allow_short=allow_short,
    )
    direction = pd.Series(position, index=df.index, name="direction")

    # ================================================================
    # 5. Strength calculation (direction * vol_scalar * vol_ratio)
    # ================================================================
    # Shift vol_scalar to prevent lookahead
    vol_scalar_shifted: pd.Series = vol_scalar.shift(1)  # type: ignore[assignment]
    vol_ratio_shifted: pd.Series = vol_ratio.shift(1)  # type: ignore[assignment]

    strength = pd.Series(
        direction.astype(float) * vol_scalar_shifted.fillna(0) * vol_ratio_shifted.fillna(0),
        index=df.index,
        name="strength",
    )

    # ShortMode DISABLED: suppress short signals
    if config.short_mode == ShortMode.DISABLED:
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    strength = strength.fillna(0.0)

    # ================================================================
    # 6. Entry/Exit signal generation
    # ================================================================
    prev_direction = direction.shift(1).fillna(Direction.NEUTRAL.value).astype(int)

    long_entry_sig = (direction == Direction.LONG.value) & (prev_direction != Direction.LONG.value)
    short_entry_sig = (direction == Direction.SHORT.value) & (
        prev_direction != Direction.SHORT.value
    )
    entries = pd.Series(
        long_entry_sig | short_entry_sig,
        index=df.index,
        name="entries",
    )

    to_neutral = (direction == Direction.NEUTRAL.value) & (
        prev_direction != Direction.NEUTRAL.value
    )
    reversal = (direction * prev_direction) < 0
    exits = pd.Series(
        to_neutral | reversal,
        index=df.index,
        name="exits",
    )

    # Signal statistics logging
    long_entries = int(long_entry_sig.sum())
    short_entries = int(short_entry_sig.sum())
    total_exits = int(exits.sum())

    if long_entries > 0 or short_entries > 0:
        logger.info(
            "Stoch-Mom Signals | Long: {}, Short: {}, Exits: {}",
            long_entries,
            short_entries,
            total_exits,
        )

    return StrategySignals(
        entries=entries,
        exits=exits,
        direction=direction,
        strength=strength,
    )
