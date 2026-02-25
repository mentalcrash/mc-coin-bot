"""Z-Momentum (MACD-V) 시그널 생성.

MACD-V (ATR-정규화 MACD) histogram + flat zone + momentum confirmation.
flat_zone 내 MACD-V histogram은 중립(0)으로 처리하여 noisy crossover를 필터링.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals
from src.strategy.z_mom.config import ShortMode

if TYPE_CHECKING:
    from src.strategy.z_mom.config import ZMomConfig


def generate_signals(df: pd.DataFrame, config: ZMomConfig) -> StrategySignals:
    """Z-Momentum (MACD-V) 시그널 생성.

    Signal Logic:
        1. MACD-V histogram > +flat_zone AND mom_return > 0 → Long
        2. MACD-V histogram < -flat_zone AND mom_return < 0 → Short
        3. |MACD-V histogram| <= flat_zone → Flat (중립)

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    # --- Shift(1): 전봉 기준 시그널 ---
    macd_v_hist = df["macd_v_hist"].shift(1)
    mom_return = df["mom_return"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    # MACD-V histogram이 flat zone 밖이고 momentum과 방향 일치
    long_signal = (macd_v_hist > config.flat_zone) & (mom_return > 0)
    short_signal = (macd_v_hist < -config.flat_zone) & (mom_return < 0)

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength: direction * vol_scalar ---
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
    config: ZMomConfig,
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
        raw = np.where(
            long_signal,
            1,
            np.where(short_signal, -1, 0),
        )

    return pd.Series(raw, index=df.index, dtype=int)
