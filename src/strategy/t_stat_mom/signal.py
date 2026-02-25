"""T-Stat Momentum 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.

Multi-lookback t-stat blend로 통계적 유의성 판단 후,
tanh(blend * scale)로 연속적 포지션 사이징.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.t_stat_mom.config import TStatMomConfig


def generate_signals(df: pd.DataFrame, config: TStatMomConfig) -> StrategySignals:
    """T-Stat Momentum 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.t_stat_mom.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    t_blend = df["t_stat_blend"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    # tanh로 연속적 conviction 계산 (양/음 대칭)
    conviction = np.tanh(t_blend * config.tanh_scale)

    # Long: t_blend > entry_threshold
    long_signal = t_blend > config.entry_threshold
    # Short: t_blend < -entry_threshold
    short_signal = t_blend < -config.entry_threshold
    # Exit: |t_blend| < exit_threshold (중립 구간)
    exit_zone = t_blend.abs() < config.exit_threshold

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        exit_zone=exit_zone,
        df=df,
        config=config,
    )

    # --- Strength: direction * vol_scalar * |conviction| ---
    abs_conviction = conviction.abs().fillna(0)
    strength = direction.astype(float) * vol_scalar.fillna(0) * abs_conviction

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
    exit_zone: pd.Series,
    df: pd.DataFrame,
    config: TStatMomConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.t_stat_mom.config import ShortMode

    if config.short_mode == ShortMode.DISABLED:
        raw = np.where(long_signal, 1, 0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        drawdown_val = df["drawdown"].shift(1)
        hedge_active = drawdown_val < config.hedge_threshold
        raw = np.where(
            long_signal,
            1,
            np.where(short_signal & hedge_active, -1, 0),
        )

    else:  # FULL
        raw = np.where(long_signal, 1, np.where(short_signal, -1, 0))

    # exit_zone에서는 중립으로 전환
    raw = np.where(exit_zone, 0, raw)

    return pd.Series(raw, index=df.index, dtype=int)
