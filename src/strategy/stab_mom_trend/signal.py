"""Stablecoin Momentum Trend 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.stab_mom_trend.config import StabMomTrendConfig


def generate_signals(df: pd.DataFrame, config: StabMomTrendConfig) -> StrategySignals:
    """Stablecoin Momentum Trend 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.stab_mom_trend.config import ShortMode

    close: pd.Series = df["close"]  # type: ignore[assignment]

    # --- Shift(1): 전봉 기준 시그널 ---
    stab_z = df["stab_z"].shift(1)
    ema_fast = df["ema_fast"].shift(1)
    ema_slow = df["ema_slow"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Trend Conditions ---
    trend_up = (close > ema_fast) & (ema_fast > ema_slow)
    trend_down = (close < ema_fast) & (ema_fast < ema_slow)

    # --- Stablecoin Conditions ---
    stab_bullish = stab_z > config.stab_long_threshold
    stab_bearish = stab_z < config.stab_short_threshold

    # --- Combined Signals ---
    long_signal = stab_bullish & trend_up
    short_signal = stab_bearish & trend_down

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        config=config,
        index=df.index,
    )

    # --- Strength ---
    base_scalar = vol_scalar.fillna(0)
    strength = direction.astype(float) * base_scalar

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
    config: StabMomTrendConfig,
    index: pd.Index,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.stab_mom_trend.config import ShortMode

    if config.short_mode == ShortMode.DISABLED:
        raw = np.where(long_signal, 1, 0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        raw = np.where(long_signal, 1, np.where(short_signal, -1, 0))

    else:  # FULL
        raw = np.where(long_signal, 1, np.where(short_signal, -1, 0))

    return pd.Series(raw, index=index, dtype=int)
