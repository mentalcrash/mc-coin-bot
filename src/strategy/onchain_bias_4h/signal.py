"""On-chain Bias 4H 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
On-chain phase gate + 4H momentum timing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.onchain_bias_4h.config import OnchainBias4hConfig


def generate_signals(df: pd.DataFrame, config: OnchainBias4hConfig) -> StrategySignals:
    """On-chain Bias 4H 시그널 생성.

    ACCUMULATION phase → LONG only
    DISTRIBUTION phase → SHORT only (HEDGE_ONLY)
    NEUTRAL → momentum follow
    Entry: er > er_min AND |price_roc| > roc_threshold AND direction 일치
    Exit: phase 전환

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.onchain_bias_4h.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    phase = df["phase"].shift(1)
    er = df["er"].shift(1)
    price_roc = df["price_roc"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Momentum Conditions ---
    momentum_ok = er > config.er_min
    strong_move = price_roc.abs() > config.roc_threshold
    roc_sign = np.sign(price_roc)

    # --- Direction based on phase + momentum ---
    # ACCUMULATION (phase=1): LONG if positive momentum
    # DISTRIBUTION (phase=-1): SHORT if negative momentum
    # NEUTRAL (phase=0): follow momentum direction
    accum_long = (phase == 1) & momentum_ok & strong_move & (roc_sign > 0)
    distrib_short = (phase == -1) & momentum_ok & strong_move & (roc_sign < 0)
    neutral_long = (phase == 0) & momentum_ok & strong_move & (roc_sign > 0)
    neutral_short = (phase == 0) & momentum_ok & strong_move & (roc_sign < 0)

    raw_direction = np.where(
        accum_long | neutral_long,
        1,
        np.where(distrib_short | neutral_short, -1, 0),
    )

    # Apply ShortMode
    if config.short_mode == ShortMode.DISABLED:
        raw_direction = np.where(raw_direction == -1, 0, raw_direction)

    direction = pd.Series(raw_direction, index=df.index, dtype=int)

    # --- Strength ---
    base_scalar = vol_scalar.fillna(0)
    strength = direction.astype(float) * base_scalar
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
