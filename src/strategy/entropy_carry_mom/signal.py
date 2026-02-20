"""Entropy-Carry-Momentum 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.

Signal Logic:
    1. Entropy percentile rank로 regime 분류:
       - low entropy (rank < low_pct) -> momentum 우위
       - high entropy (rank > high_pct) -> carry 우위
       - 중간 -> 균등 가중
    2. Momentum signal: mom_direction 기반
    3. Carry signal: -sign(avg_FR) 기반 (positive FR -> short, negative -> long)
    4. Final direction: entropy-adaptive weighting으로 결합
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.entropy_carry_mom.config import EntropyCarryMomConfig


def generate_signals(df: pd.DataFrame, config: EntropyCarryMomConfig) -> StrategySignals:
    """Entropy-Carry-Momentum 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.entropy_carry_mom.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    entropy_rank = df["entropy_rank"].shift(1)
    mom_direction = df["mom_direction"].shift(1)
    mom_strength = df["mom_strength"].shift(1)
    avg_fr = df["avg_funding_rate"].shift(1)
    fr_zscore = df["fr_zscore"].shift(1)
    carry_direction = df["carry_direction"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Entropy Regime Classification ---
    low_pct = config.entropy_low_pct / 100.0
    high_pct = config.entropy_high_pct / 100.0

    is_low_entropy = entropy_rank < low_pct
    is_high_entropy = entropy_rank > high_pct

    # --- Adaptive Weights ---
    # Low entropy: momentum dominates
    # High entropy: carry dominates
    # Mid: equal weighting (0.5 / 0.5)
    mom_w = config.mom_weight_low_entropy
    carry_w = config.carry_weight_high_entropy

    w_mom = pd.Series(
        np.where(
            is_low_entropy,
            mom_w,
            np.where(is_high_entropy, 1.0 - carry_w, 0.5),
        ),
        index=df.index,
    )
    w_carry = pd.Series(
        np.where(
            is_high_entropy,
            carry_w,
            np.where(is_low_entropy, 1.0 - mom_w, 0.5),
        ),
        index=df.index,
    )

    # --- Momentum Score: direction * min(strength, 1) ---
    mom_score = mom_direction.fillna(0) * mom_strength.fillna(0).clip(upper=1.0)

    # --- Carry Score: carry_direction * |fr_zscore| clamped ---
    fr_active = avg_fr.abs() > config.fr_entry_threshold
    fr_conviction = fr_zscore.abs().clip(upper=3.0).fillna(0) / 3.0
    carry_score = carry_direction.fillna(0) * fr_conviction * fr_active.astype(float)

    # --- Combined Score ---
    combined_score = w_mom * mom_score + w_carry * carry_score

    # --- Direction from combined score ---
    raw_direction = np.sign(combined_score).fillna(0).astype(int)

    # --- Apply ShortMode ---
    long_signal = raw_direction > 0
    short_signal = raw_direction < 0

    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength ---
    # Conviction: absolute combined score (already 0~1 range from components)
    conviction = combined_score.abs().clip(lower=0.2, upper=1.0).fillna(0)
    strength = direction.astype(float) * vol_scalar.fillna(0) * conviction

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
    config: EntropyCarryMomConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.entropy_carry_mom.config import ShortMode

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
