"""Fractal-Filtered Momentum 시그널 생성.

D < threshold → deterministic regime → trend following active.
Shift(1) Rule: 모든 feature에 shift(1) 적용.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.fractal_mom.config import ShortMode
from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.fractal_mom.config import FractalMomConfig


def generate_signals(df: pd.DataFrame, config: FractalMomConfig) -> StrategySignals:
    """Fractal-Filtered Momentum 시그널 생성.

    Signal Logic:
        1. fractal_dim < threshold → deterministic (trend) regime
        2. ER > er_threshold → trend 확인
        3. Fast momentum > 0 → long, < 0 → short
        4. 두 조건 동시 충족 시에만 시그널 활성화

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    # --- Shift(1): 전봉 기준 ---
    fractal_dim = df["fractal_dim"].shift(1)
    mom_fast = df["mom_fast"].shift(1)
    mom_slow = df["mom_slow"].shift(1)
    er = df["er"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Regime Filter ---
    # D < threshold = deterministic (persistent) regime
    deterministic = fractal_dim < config.fractal_threshold
    # ER confirms trend quality
    trend_confirmed = er > config.er_threshold
    # Regime active: both conditions
    regime_active = deterministic & trend_confirmed

    # --- Momentum Direction ---
    # Fast-slow crossover: mom_fast와 mom_slow 방향 일치
    fast_up = mom_fast > 0
    slow_up = mom_slow > 0
    fast_down = mom_fast < 0
    slow_down = mom_slow < 0

    long_signal = regime_active & fast_up & slow_up
    short_signal = regime_active & fast_down & slow_down

    # --- Direction (ShortMode 3-way) ---
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
    config: FractalMomConfig,
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
