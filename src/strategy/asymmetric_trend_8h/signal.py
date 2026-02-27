"""Directional-Asymmetric Multi-Scale Momentum 시그널 생성.

비대칭 lookback: UP은 느리게 확인(15/30/63), DOWN은 빠르게 반응(3/6/15).
3-scale consensus 투표로 방향 결정.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.asymmetric_trend_8h.config import AsymmetricTrend8hConfig


def generate_signals(df: pd.DataFrame, config: AsymmetricTrend8hConfig) -> StrategySignals:
    """Directional-Asymmetric Multi-Scale Momentum 시그널 생성.

    Signal Logic:
        1. UP ROC 3-scale sign 평균 → up_score (slow confirmation)
        2. DN ROC 3-scale sign 평균 → dn_score (fast reaction)
        3. up_score >= consensus_threshold → long_signal
        4. dn_score <= -consensus_threshold → short_signal
        5. strength = direction * relevant_score_magnitude * vol_scalar

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.asymmetric_trend_8h.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    up_s = df["up_roc_short"].shift(1)
    up_m = df["up_roc_mid"].shift(1)
    up_l = df["up_roc_long"].shift(1)
    dn_s = df["dn_roc_short"].shift(1)
    dn_m = df["dn_roc_mid"].shift(1)
    dn_l = df["dn_roc_long"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)
    dd = df["drawdown"].shift(1)

    # --- Up-momentum: slow confirmation (longer lookbacks) ---
    # Score = mean of sign(each lookback ROC)
    up_score: pd.Series = (np.sign(up_s) + np.sign(up_m) + np.sign(up_l)) / 3.0  # type: ignore[assignment]

    # --- Down-momentum: fast reaction (shorter lookbacks) ---
    dn_score: pd.Series = (np.sign(dn_s) + np.sign(dn_m) + np.sign(dn_l)) / 3.0  # type: ignore[assignment]

    # --- Direction from consensus threshold ---
    long_signal = up_score >= config.consensus_threshold  # 2/3 up scales agree
    short_signal = dn_score <= -config.consensus_threshold  # 2/3 down scales agree

    # --- Apply ShortMode ---
    dd_series: pd.Series = dd  # type: ignore[assignment]
    direction = _compute_direction(long_signal, short_signal, dd_series, config)

    # --- Strength: use the relevant score magnitude ---
    up_strength: pd.Series = up_score.abs()  # type: ignore[assignment]
    dn_strength: pd.Series = dn_score.abs()  # type: ignore[assignment]
    raw_strength = np.where(
        direction == 1,
        up_strength,
        np.where(direction == -1, dn_strength, 0.0),
    )
    strength = direction.astype(float) * raw_strength * vol_scalar.fillna(0)

    # --- HEDGE_ONLY strength reduction for shorts ---
    if config.short_mode == ShortMode.HEDGE_ONLY:
        strength = pd.Series(
            np.where(direction == -1, strength * config.hedge_strength_ratio, strength),
            index=df.index,
        )

    strength = pd.Series(strength, index=df.index).fillna(0.0)

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
    drawdown_series: pd.Series,
    config: AsymmetricTrend8hConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.asymmetric_trend_8h.config import ShortMode

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

    return pd.Series(raw, index=long_signal.index, dtype=int)
