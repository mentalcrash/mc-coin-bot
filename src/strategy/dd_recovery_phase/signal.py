"""Drawdown-Recovery Phase 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.dd_recovery_phase.config import DDRecoveryPhaseConfig


def generate_signals(df: pd.DataFrame, config: DDRecoveryPhaseConfig) -> StrategySignals:
    """Drawdown-Recovery Phase 시그널 생성.

    Long 조건: 깊은 drawdown 경험 후 recovery_ratio 이상 회복 + momentum 상승.
    Short 조건: 현재 drawdown 깊어지고 + momentum 하락 (HEDGE_ONLY).

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.dd_recovery_phase.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    recovery_val = df["recovery_ratio_val"].shift(1)
    was_deep = df["was_deep_dd"].shift(1)
    momentum = df["momentum"].shift(1)
    dd = df["drawdown"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    # Long: 깊은 drawdown 구간을 거쳤고 + 회복 비율 충족 + momentum 양
    long_signal = (was_deep == 1) & (recovery_val >= config.recovery_ratio) & (momentum > 0)

    # Short: 현재 drawdown가 깊고 momentum 하락
    short_signal = (dd <= config.dd_threshold) & (momentum < 0)

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
    config: DDRecoveryPhaseConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.dd_recovery_phase.config import ShortMode

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
