"""VRP-Trend 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
고VRP(IV>>RV) + 상승추세 → Long, 저VRP(IV<<RV) + 하락추세 → Short.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.vrp_trend.config import VrpTrendConfig


def generate_signals(df: pd.DataFrame, config: VrpTrendConfig) -> StrategySignals:
    """VRP-Trend 시그널 생성.

    Signal Logic:
        - vrp_zscore >= vrp_entry_z AND above_trend → Long (고VRP + 상승추세)
        - vrp_zscore <= vrp_exit_z AND NOT above_trend → Short (저VRP + 하락추세)
        - else → Neutral

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.vrp_trend.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    vrp_zscore = df["vrp_zscore"].shift(1)
    above_trend = df["above_trend"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    # 고VRP + 상승추세 = 시장 과공포이나 실제 트렌드 유지 → Long
    long_signal = (vrp_zscore >= config.vrp_entry_z) & (above_trend == 1)
    # 저VRP + 하락추세 = IV 과소평가 + 하락 = 실제 리스크 → Short
    short_signal = (vrp_zscore <= config.vrp_exit_z) & (above_trend == 0)

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
    config: VrpTrendConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.vrp_trend.config import ShortMode

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
