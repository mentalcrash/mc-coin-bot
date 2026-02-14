"""Variance Decomposition Momentum 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
good_var 지배적(var_ratio > threshold) + 가격 모멘텀 양수 → Long.
bad_var 지배적(var_ratio < 1-threshold) + 가격 모멘텀 음수 → Short.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.vardecomp_mom.config import VardecompMomConfig


def generate_signals(df: pd.DataFrame, config: VardecompMomConfig) -> StrategySignals:
    """Variance Decomposition Momentum 시그널 생성.

    Signal Logic:
        - var_ratio > threshold AND price_mom > 0 → Long (건강한 상방 추세)
        - var_ratio < (1 - threshold) AND price_mom < 0 → Short (건강한 하방 추세)
        - Otherwise → Neutral

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.vardecomp_mom.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    var_ratio = df["var_ratio"].shift(1)
    price_mom = df["price_mom"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    # good_var 지배적 + 상승 모멘텀 = long
    long_signal = (var_ratio > config.var_ratio_threshold) & (price_mom > 0)
    # bad_var 지배적 + 하락 모멘텀 = short
    short_threshold = 1.0 - config.var_ratio_threshold
    short_signal = (var_ratio < short_threshold) & (price_mom < 0)

    # --- Direction (ShortMode 분기) ---
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
    config: VardecompMomConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.vardecomp_mom.config import ShortMode

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
