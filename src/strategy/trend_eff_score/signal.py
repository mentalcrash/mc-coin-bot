"""Trend Efficiency Scorer 시그널 생성.

ER 필터 + 다중 ROC 합의(scoring)로 방향 결정.
Shift(1) Rule: 모든 feature에 shift(1) 적용.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.trend_eff_score.config import ShortMode
from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.trend_eff_score.config import TrendEffScoreConfig


def generate_signals(df: pd.DataFrame, config: TrendEffScoreConfig) -> StrategySignals:
    """Trend Efficiency Scorer 시그널 생성.

    Signal Logic:
        1. is_trending = ER > er_threshold
        2. mom_score = sign(roc_short) + sign(roc_medium) + sign(roc_long)
        3. direction = sign(mom_score) if |mom_score| >= min_score else 0
        4. direction *= is_trending (ER OFF -> flat)

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    # --- Shift(1): 전봉 기준 ---
    er = df["er"].shift(1)
    roc_s = df["roc_short"].shift(1)
    roc_m = df["roc_medium"].shift(1)
    roc_l = df["roc_long"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    # 1. Trend quality filter
    is_trending = er > config.er_threshold

    # 2. Multi-horizon momentum score (-3 ~ +3)
    mom_score = np.sign(roc_s) + np.sign(roc_m) + np.sign(roc_l)

    # 3. Direction from consensus (minimum agreement threshold)
    has_consensus = np.abs(mom_score) >= config.min_score
    raw_dir = np.sign(mom_score)
    consensus_dir = pd.Series(
        np.where(has_consensus, raw_dir, 0),
        index=df.index,
    )

    # 4. Apply trend quality gate
    gated_dir = pd.Series(
        np.where(is_trending, consensus_dir, 0),
        index=df.index,
        dtype=int,
    )

    long_signal = gated_dir > 0
    short_signal = gated_dir < 0

    # --- Direction (ShortMode 3-way) ---
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
    config: TrendEffScoreConfig,
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
        raw = np.where(long_signal, 1, np.where(short_signal, -1, 0))

    return pd.Series(raw, index=df.index, dtype=int)
