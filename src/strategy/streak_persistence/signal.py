"""Return Streak Persistence 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.streak_persistence.config import StreakPersistenceConfig


def generate_signals(df: pd.DataFrame, config: StreakPersistenceConfig) -> StrategySignals:
    """Return Streak Persistence 시그널 생성.

    Long: positive streak >= threshold + momentum > 0.
    Short: negative streak >= threshold + momentum < 0.
    Streak conviction: min(streak, max_cap) / max_cap 으로 strength 가중.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.streak_persistence.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    pos_streak = df["positive_streak"].shift(1)
    neg_streak = df["negative_streak"].shift(1)
    momentum = df["momentum"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    # Long: positive streak >= threshold + positive momentum
    long_signal = (pos_streak >= config.streak_threshold) & (momentum > 0)
    # Short: negative streak >= threshold + negative momentum
    short_signal = (neg_streak >= config.streak_threshold) & (momentum < 0)

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Streak conviction: capped streak normalized ---
    capped_pos = pos_streak.clip(upper=config.max_streak_cap)
    capped_neg = neg_streak.clip(upper=config.max_streak_cap)
    conviction = pd.Series(
        np.where(
            direction == 1,
            capped_pos / config.max_streak_cap,
            np.where(direction == -1, capped_neg / config.max_streak_cap, 0.0),
        ),
        index=df.index,
    )

    # --- Strength ---
    strength = direction.astype(float) * vol_scalar.fillna(0) * conviction.clip(lower=0.3)

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
    config: StreakPersistenceConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.streak_persistence.config import ShortMode

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
