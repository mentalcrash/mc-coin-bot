"""HMM Regime Signal Generator.

Regime 기반 시그널 생성.
HMM이 분류한 Bull/Bear/Sideways regime에 따라 포지션 방향과 강도를 결정합니다.

Signal Formula:
    1. regime/prob/vol_scalar에 shift(1) 적용 (lookahead prevention)
    2. Bull(1) -> LONG, Bear(-1) -> SHORT, Sideways(0)/Unknown(-1) -> FLAT
    3. strength = direction * prob * vol_scalar

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: lookahead bias prevention
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.hmm_regime.config import ShortMode
from src.strategy.types import Direction, StrategySignals

if TYPE_CHECKING:
    from src.strategy.hmm_regime.config import HMMRegimeConfig


def generate_signals(
    df: pd.DataFrame,
    config: HMMRegimeConfig | None = None,
) -> StrategySignals:
    """HMM Regime signal generation.

    Args:
        df: Preprocessed DataFrame (preprocess() output)
            Required columns: regime, regime_prob, vol_scalar
        config: HMM Regime config. Defaults used if None.

    Returns:
        StrategySignals NamedTuple:
            - entries: Entry signals (bool Series)
            - exits: Exit signals (bool Series)
            - direction: Direction series (-1, 0, 1)
            - strength: Signal strength (unbounded)

    Raises:
        ValueError: If required columns are missing
    """
    if config is None:
        from src.strategy.hmm_regime.config import HMMRegimeConfig

        config = HMMRegimeConfig()

    # Input validation
    required_cols = {"regime", "regime_prob", "vol_scalar"}
    if config.short_mode == ShortMode.HEDGE_ONLY:
        required_cols.add("drawdown")
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # Shift(1) applied - use previous bar data
    regime_prev: pd.Series = df["regime"].shift(1)  # type: ignore[assignment]
    prob_prev: pd.Series = df["regime_prob"].shift(1)  # type: ignore[assignment]
    vol_scalar_prev: pd.Series = df["vol_scalar"].shift(1)  # type: ignore[assignment]

    # Bull(1) -> LONG, Bear(-1) -> SHORT, Sideways(0) or Unknown(-1 in regime) -> FLAT
    direction_raw = np.where(
        regime_prev == 1,
        1,
        np.where(regime_prev == -1, -1, 0),
    )

    # Strength weighted by probability and vol_scalar
    strength_raw = pd.Series(
        direction_raw * prob_prev * vol_scalar_prev,
        index=df.index,
    )

    direction_sign = pd.Series(np.sign(strength_raw), index=df.index)
    direction = pd.Series(
        direction_sign.fillna(0).astype(int),
        index=df.index,
        name="direction",
    )
    strength = pd.Series(
        strength_raw.fillna(0),
        index=df.index,
        name="strength",
    )

    # ShortMode processing
    if config.short_mode == ShortMode.DISABLED:
        # Long-Only: convert all short signals to neutral
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        # Hedge mode: allow shorts only when drawdown exceeds threshold
        drawdown_series: pd.Series = df["drawdown"]  # type: ignore[assignment]
        hedge_active = drawdown_series < config.hedge_threshold

        # Suppress shorts when hedge is inactive
        short_mask = direction == Direction.SHORT
        suppress_short = short_mask & ~hedge_active
        direction = direction.where(~suppress_short, Direction.NEUTRAL)
        strength = strength.where(~suppress_short, 0.0)

        # Scale strength for active hedge shorts
        active_short = short_mask & hedge_active
        strength = strength.where(
            ~active_short,
            strength * config.hedge_strength_ratio,
        )

        # Hedge activation stats logging
        hedge_days = int(hedge_active.sum())
        if hedge_days > 0:
            logger.info(
                "Hedge Mode | Active: {} days ({:.1f}%), Threshold: {:.1f}%",
                hedge_days,
                hedge_days / len(hedge_active) * 100,
                config.hedge_threshold * 100,
            )

    # else: ShortMode.FULL - keep all signals as-is

    # Entry/Exit signals
    prev_direction = direction.shift(1).fillna(0)

    # Long entry
    long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)
    # Short entry
    short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)

    entries = pd.Series(
        long_entry | short_entry,
        index=df.index,
        name="entries",
    )

    # Exit: to neutral or reversal
    to_neutral = (direction == Direction.NEUTRAL) & (prev_direction != Direction.NEUTRAL)
    reversal = direction * prev_direction < 0

    exits = pd.Series(
        to_neutral | reversal,
        index=df.index,
        name="exits",
    )

    # Log stats
    valid_strength = strength[strength != 0]
    long_signals = strength[strength > 0]
    short_signals = strength[strength < 0]

    if len(valid_strength) > 0:
        logger.info(
            "HMM Signals | Total: {} signals, Long: {} ({:.1f}%), Short: {} ({:.1f}%)",
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
