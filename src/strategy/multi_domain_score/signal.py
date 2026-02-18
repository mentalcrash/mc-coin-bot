"""Multi-Domain Score 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
4차원 soft scoring → composite threshold 기반 bar-by-bar 판정.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.multi_domain_score.config import MultiDomainScoreConfig


def generate_signals(df: pd.DataFrame, config: MultiDomainScoreConfig) -> StrategySignals:
    """Multi-Domain Score 시그널 생성.

    D1 trend_score: sign(close - sma) * min(er / 0.5, 1.0)
    D2 volume_score: sign(obv_slope) * min(|obv_slope| / threshold, 1.0)
    D3 deriv_score: -sign(fr_z) * min(|fr_z| / cap, 1.0)
    D4 vol_score: sign(price_dir) * min(rv_ratio / 1.5, 1.0) if expanding
    composite: w1*D1 + w2*D2 + w3*D3 + w4*D4
    Entry: |composite| > entry_threshold

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.multi_domain_score.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    er = df["er"].shift(1)
    sma_dir = df["sma_direction"].shift(1)
    obv_slope = df["obv_slope"].shift(1)
    fr_z = df["fr_z"].shift(1)
    rv_ratio = df["rv_ratio"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)
    returns = df["returns"].shift(1)

    # --- D1: Trend Score [-1, 1] ---
    er_norm = (er / 0.5).clip(upper=1.0).fillna(0)
    trend_score = sma_dir.fillna(0) * er_norm

    # --- D2: Volume Score [-1, 1] ---
    obv_abs = obv_slope.abs()
    # Use median of obv_abs as normalization threshold
    obv_threshold = obv_abs.rolling(42, min_periods=10).median().fillna(obv_abs.median())
    obv_norm = (obv_abs / obv_threshold.replace(0, float("nan"))).clip(upper=1.0).fillna(0)
    volume_score = np.sign(obv_slope.fillna(0)) * obv_norm

    # --- D3: Derivatives Score [-1, 1] ---
    fr_norm = (fr_z.abs() / config.fr_score_cap).clip(upper=1.0).fillna(0)
    deriv_score = -np.sign(fr_z.fillna(0)) * fr_norm  # Contrarian

    # --- D4: Volatility Score [-1, 1] ---
    price_dir = np.sign(returns.fillna(0))
    expanding = rv_ratio > 1.0
    rv_norm = (rv_ratio / 1.5).clip(upper=1.0).fillna(0)
    vol_score_expanding = price_dir * rv_norm
    vol_score_contracting = price_dir * rv_norm * 0.5
    vol_score = vol_score_expanding.where(expanding, vol_score_contracting)

    # --- Composite Score ---
    composite = (
        config.w_trend * trend_score
        + config.w_volume * volume_score
        + config.w_derivatives * deriv_score
        + config.w_volatility * vol_score
    )

    # --- Direction ---
    raw_direction = np.where(
        composite.abs() > config.entry_threshold,
        np.sign(composite),
        0,
    )

    # Apply ShortMode
    if config.short_mode == ShortMode.DISABLED:
        raw_direction = np.where(raw_direction == -1, 0, raw_direction)

    direction = pd.Series(raw_direction, index=df.index, dtype=int)

    # --- Strength ---
    base_scalar = vol_scalar.fillna(0) * composite.abs().clip(upper=1.0)
    strength = direction.astype(float) * base_scalar
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
