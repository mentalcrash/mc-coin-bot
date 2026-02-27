"""Funding Rate Event Trigger + 12H Momentum Context 시그널 생성 (8H TF).

FR z-score 극단값 + EMA 추세 컨텍스트 기반 이벤트 트리거 시그널.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.fr_event_mtf_8h.config import FrEventMtf8hConfig


def generate_signals(df: pd.DataFrame, config: FrEventMtf8hConfig) -> StrategySignals:
    """FR Event MTF 8H 시그널 생성.

    Signal Logic:
        1. FR z-score < -threshold & EMA bull → long (shorts crowded + uptrend)
        2. FR z-score > +threshold & EMA bear → short (longs crowded + downtrend)
        3. min_hold_bars ffill로 포지션 유지 (whipsaw 방지)
        4. ShortMode 분기 적용
        5. strength = |direction| * vol_scalar * clipped_fr_z / 3.0

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.fr_event_mtf_8h.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    fr_z = df["fr_zscore"].shift(1)
    ema_f = df["ema_fast"].shift(1)
    ema_s = df["ema_slow"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)
    dd = df["drawdown"].shift(1)

    # --- Trend context ---
    trend_bull = ema_f > ema_s
    trend_bear = ema_f < ema_s

    # --- Event triggers ---
    # Shorts crowded (negative FR z-score) + uptrend → long
    long_event = (fr_z < -config.fr_extreme_threshold) & trend_bull
    # Longs crowded (positive FR z-score) + downtrend → short
    short_event = (fr_z > config.fr_extreme_threshold) & trend_bear

    # --- Raw signal ---
    raw_signal = np.where(long_event, 1, np.where(short_event, -1, 0))

    # --- Min hold enforcement (vectorized) ---
    # Replace 0 with NaN, ffill with limit, fill remaining NaN with 0
    signal_series = pd.Series(raw_signal, index=df.index).replace(0, np.nan)
    held_signal = signal_series.ffill(limit=config.min_hold_bars).fillna(0).astype(int)

    # --- Direction (ShortMode 분기) ---
    dd_series: pd.Series = dd  # type: ignore[assignment]
    direction = _compute_direction(held_signal, dd_series, config)

    # --- Strength: |direction| * vol_scalar * clipped_fr_z / 3.0 ---
    strength = (
        direction.astype(float).abs() * vol_scalar.fillna(0) * fr_z.abs().clip(upper=3.0) / 3.0
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
    held_signal: pd.Series,
    drawdown_series: pd.Series,
    config: FrEventMtf8hConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.fr_event_mtf_8h.config import ShortMode

    long_signal = held_signal > 0
    short_signal = held_signal < 0

    if config.short_mode == ShortMode.DISABLED:
        raw = np.where(long_signal, 1, 0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        hedge_active = drawdown_series < config.hedge_threshold
        raw = np.where(
            long_signal,
            1,
            np.where(short_signal & hedge_active, -1, 0),
        )

    else:  # FULL
        raw = np.where(long_signal, 1, np.where(short_signal, -1, 0))

    return pd.Series(raw, index=held_signal.index, dtype=int)
