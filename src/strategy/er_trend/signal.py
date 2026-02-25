"""ER Trend 시그널 생성.

Multi-lookback Signed ER의 가중 합성으로 추세 방향과 품질을 동시에 판단.
Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.er_trend.config import ShortMode
from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.er_trend.config import ErTrendConfig


def generate_signals(df: pd.DataFrame, config: ErTrendConfig) -> StrategySignals:
    """ER Trend 시그널 생성.

    Signal Logic:
        1. composite_ser = w_fast*SER_fast + w_mid*SER_mid + w_slow*SER_slow
        2. long: composite_ser > entry_threshold
        3. short: composite_ser < -entry_threshold
        4. flat: |composite_ser| <= entry_threshold
        5. conviction = |composite_ser| (방향은 sign으로, 크기는 conviction으로)

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    # --- Shift(1): 전봉 기준 시그널 ---
    composite = df["composite_ser"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    long_signal = composite > config.entry_threshold
    short_signal = composite < -config.entry_threshold

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength: direction * vol_scalar * |composite| as conviction ---
    conviction = composite.abs().fillna(0).clip(upper=1.0)
    strength = direction.astype(float) * vol_scalar.fillna(0) * conviction

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
    config: ErTrendConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
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
