"""Stablecoin Composition Shift Signal Generation.

Direction logic:
    - share_roc_short > 0 AND share_roc_long > 0 → +1 (USDT dominance rising = risk-on)
    - share_roc_short < 0 AND share_roc_long < 0 → -1 (USDC dominance rising = cautious)
    - Otherwise → 0 (mixed signal)

Strength = direction * vol_scalar
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.models.types import Direction
from src.strategy.stab_comp.config import StabCompConfig
from src.strategy.tsmom.config import ShortMode
from src.strategy.types import StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: StabCompConfig | None = None,
) -> StrategySignals:
    """7D/30D ROC 방향 기반 시그널 생성.

    Args:
        df: preprocess() 결과 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    if config is None:
        config = StabCompConfig()

    required = {"vol_scalar"}
    missing = required - set(df.columns)
    if missing:
        msg = f"Missing columns: {missing}"
        raise ValueError(msg)

    vol_scalar: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    # Composition ROC scores
    if "share_roc_short" in df.columns and "share_roc_long" in df.columns:
        roc_short: pd.Series = df["share_roc_short"].fillna(0.0)  # type: ignore[assignment]
        roc_long: pd.Series = df["share_roc_long"].fillna(0.0)  # type: ignore[assignment]

        raw_dir = pd.Series(
            np.where(
                (roc_short > 0) & (roc_long > 0),
                1.0,
                np.where((roc_short < 0) & (roc_long < 0), -1.0, 0.0),
            ),
            index=df.index,
        )
    else:
        raw_dir = pd.Series(0.0, index=df.index)

    # Strength: direction * vol_scalar
    raw_strength = raw_dir * vol_scalar

    # Shift(1) — lookahead bias 방지
    signal_shifted = raw_strength.shift(1).fillna(0.0)

    direction = pd.Series(
        np.sign(signal_shifted).fillna(0).astype(int),
        index=df.index,
        name="direction",
    )
    strength = pd.Series(signal_shifted, index=df.index, name="strength")

    # ShortMode 처리
    if config.short_mode in {ShortMode.DISABLED, ShortMode.HEDGE_ONLY}:
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    # Entry / Exit
    prev_dir = direction.shift(1).fillna(0)
    long_entry = (direction == Direction.LONG) & (prev_dir != Direction.LONG)
    short_entry = (direction == Direction.SHORT) & (prev_dir != Direction.SHORT)
    entries = pd.Series(long_entry | short_entry, index=df.index, name="entries")

    to_neutral = (direction == Direction.NEUTRAL) & (prev_dir != Direction.NEUTRAL)
    reversal = direction * prev_dir < 0
    exits = pd.Series(to_neutral | reversal, index=df.index, name="exits")

    n_total = len(df)
    n_long = int((direction == Direction.LONG).sum())
    n_short = int((direction == Direction.SHORT).sum())
    logger.info(
        "Stab-Comp signals | Total: {}, Long: {} ({:.1%}), Short: {} ({:.1%})",
        n_total,
        n_long,
        n_long / max(n_total, 1),
        n_short,
        n_short / max(n_total, 1),
    )

    return StrategySignals(entries=entries, exits=exits, direction=direction, strength=strength)
