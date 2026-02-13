"""Anchored Trend-Following 3H 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
Anchor-Mom과 동일 로직, 3H TF 적응.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.atf_3h.config import Atf3hConfig


def generate_signals(df: pd.DataFrame, config: Atf3hConfig) -> StrategySignals:
    """Anchored Trend-Following 3H 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.atf_3h.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    nearness = df["nearness"].shift(1)
    mom_dir = df["mom_direction"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    strong_long = (nearness > config.strong_nearness) & (mom_dir > 0)
    weak_long = (nearness > config.weak_nearness) & (mom_dir > 0) & ~strong_long
    short_signal = (nearness < config.short_nearness) & (mom_dir < 0)

    long_signal = strong_long | weak_long

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength: strong_long gets full scalar, weak_long gets 0.7x ---
    base_scalar = vol_scalar.fillna(0)
    nearness_boost = pd.Series(
        np.where(strong_long, 1.0, np.where(weak_long, 0.7, 1.0)),
        index=df.index,
    )
    strength = direction.astype(float) * base_scalar * nearness_boost

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
    config: Atf3hConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.atf_3h.config import ShortMode

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
        raw = np.where(long_signal, 1, np.where(short_signal, -1, 0))

    return pd.Series(raw, index=df.index, dtype=int)
