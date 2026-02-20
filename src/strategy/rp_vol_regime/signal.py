"""Realized-Parkinson Vol Regime 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.

PV/RV ratio 낮음(추세 지속 레짐) -> 모멘텀 방향 진입.
PV/RV ratio 높음(축적/분배) -> 모멘텀 반대 방향 (contrarian).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.rp_vol_regime.config import RpVolRegimeConfig


def generate_signals(df: pd.DataFrame, config: RpVolRegimeConfig) -> StrategySignals:
    """Realized-Parkinson Vol Regime 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.rp_vol_regime.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    pv_rv_zscore = df["pv_rv_zscore"].shift(1)
    momentum = df["momentum"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    # Low PV/RV (trend continuation regime): follow momentum
    trend_regime = pv_rv_zscore < config.ratio_lower
    long_trend = trend_regime & (momentum > 0)
    short_trend = trend_regime & (momentum < 0)

    # High PV/RV (accumulation/distribution): follow momentum too
    # (high intraday volatility with compressed close-to-close = breakout imminent)
    accum_regime = pv_rv_zscore > config.ratio_upper
    long_accum = accum_regime & (momentum > 0)
    short_accum = accum_regime & (momentum < 0)

    long_signal = long_trend | long_accum
    short_signal = short_trend | short_accum

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
    config: RpVolRegimeConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.rp_vol_regime.config import ShortMode

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
