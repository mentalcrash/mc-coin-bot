"""Carry-Conditional Momentum 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
가격 모멘텀 + FR level agreement → 포지션 사이징 조절.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.carry_cond_mom.config import CarryCondMomConfig


def generate_signals(df: pd.DataFrame, config: CarryCondMomConfig) -> StrategySignals:
    """Carry-Conditional Momentum 시그널 생성.

    Signal Logic:
        - price_mom > 0 → base direction = Long
        - price_mom < 0 → base direction = Short (mode에 따라)
        - agreement > 0 → strength * agreement_boost (consensus)
        - agreement < 0 → strength * disagreement_penalty (conflict)

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.carry_cond_mom.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    price_mom = df["price_mom"].shift(1)
    agreement = df["agreement"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    long_signal = price_mom > 0
    short_signal = price_mom < 0

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Base Strength ---
    base_strength = direction.astype(float) * vol_scalar.fillna(0)

    # --- Agreement Conditioning ---
    # agreement > 0 (consensus): boost
    # agreement < 0 (conflict): penalize
    # agreement == 0 (neutral): no adjustment
    conditioning = pd.Series(
        np.where(
            agreement > 0,
            config.agreement_boost,
            np.where(agreement < 0, config.disagreement_penalty, 1.0),
        ),
        index=df.index,
    )
    strength = base_strength * conditioning

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
    config: CarryCondMomConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.carry_cond_mom.config import ShortMode

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
