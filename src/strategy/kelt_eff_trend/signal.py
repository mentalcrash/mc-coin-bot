"""Keltner Efficiency Trend 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
KC 돌파 + ER 확인 → 이진 결정 (conviction scalar 천장 회피).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.kelt_eff_trend.config import KeltEffTrendConfig


def generate_signals(df: pd.DataFrame, config: KeltEffTrendConfig) -> StrategySignals:
    """Keltner Efficiency Trend 시그널 생성.

    Signal Logic:
        - close > KC_upper AND ER > er_threshold → Long (상방 돌파 + 품질 확인)
        - close < KC_lower AND ER > er_threshold → Short (하방 돌파 + 품질 확인)
        - Otherwise → Neutral

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.kelt_eff_trend.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    close_prev = df["close"].shift(1)
    kc_upper = df["kc_upper"].shift(1)
    kc_lower = df["kc_lower"].shift(1)
    er = df["efficiency_ratio"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    quality_gate = er > config.er_threshold
    long_signal = (close_prev > kc_upper) & quality_gate
    short_signal = (close_prev < kc_lower) & quality_gate

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength (binary: direction * vol_scalar, no conviction scalar) ---
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
    config: KeltEffTrendConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.kelt_eff_trend.config import ShortMode

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
