"""Carry-Momentum Convergence 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.

Signal Logic:
    1. 가격 모멘텀(EMA cross)이 방향 결정 (alpha source)
    2. FR z-score magnitude가 conviction modifier
    3. Convergence score가 최종 strength를 조절
       - 수렴 시 boost (1.5x), 발산 시 penalty (0.3x)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.carry_mom_convergence.config import CarryMomConvergenceConfig


def generate_signals(df: pd.DataFrame, config: CarryMomConvergenceConfig) -> StrategySignals:
    """Carry-Momentum Convergence 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.carry_mom_convergence.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    trend_dir = df["trend_direction"].shift(1)
    price_mom = df["price_mom"].shift(1)
    fr_zscore = df["fr_zscore"].shift(1)
    convergence_score = df["convergence_score"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    # Price momentum direction from EMA cross
    long_signal = (trend_dir > 0) & (price_mom > 0)
    short_signal = (trend_dir < 0) & (price_mom < 0)

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Conviction from FR z-score ---
    # FR z-score magnitude as conviction (clamp 0~1 from range 0~3)
    fr_conviction = fr_zscore.abs().clip(upper=3.0).fillna(0) / 3.0
    # Ensure minimum conviction when active
    fr_conviction = fr_conviction.clip(lower=0.3)

    # --- Strength ---
    # direction * vol_scalar * fr_conviction * convergence_score
    strength = (
        direction.astype(float)
        * vol_scalar.fillna(0)
        * fr_conviction
        * convergence_score.fillna(1.0)
    )

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
    config: CarryMomConvergenceConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.carry_mom_convergence.config import ShortMode

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
