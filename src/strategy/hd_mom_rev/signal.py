"""Half-Day Momentum-Reversal 시그널 생성.

정상일(jump_score < threshold) -> momentum: 전반부 방향 유지
급변일(jump_score >= threshold) -> reversal: 전반부 방향 반전

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.hd_mom_rev.config import HdMomRevConfig


def generate_signals(df: pd.DataFrame, config: HdMomRevConfig) -> StrategySignals:
    """Half-Day Momentum-Reversal 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.hd_mom_rev.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    half_return_smooth = df["half_return_smooth"].shift(1)
    jump_score = df["jump_score"].shift(1)
    is_jump = df["is_jump"].shift(1).fillna(value=False).infer_objects(copy=False).astype(bool)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    # Normal day (no jump): momentum - follow the half return direction
    mom_direction = np.sign(half_return_smooth)
    # Jump day: reversal - opposite of the half return direction
    rev_direction = -np.sign(half_return_smooth)

    # Choose based on jump flag
    raw_signal_direction = pd.Series(
        np.where(is_jump, rev_direction, mom_direction),
        index=df.index,
    )

    # --- Confidence ---
    # Momentum: confidence scales with consistency (capped)
    mom_confidence = (jump_score / config.jump_threshold).clip(upper=config.confidence_cap)
    # Reversal: confidence scales with overshoot
    rev_confidence = ((jump_score - config.jump_threshold) / config.jump_threshold).clip(
        lower=0.0, upper=config.confidence_cap
    )
    confidence = pd.Series(
        np.where(is_jump, rev_confidence, mom_confidence),
        index=df.index,
    ).fillna(0)

    # --- Long/Short signals from raw direction ---
    long_signal = raw_signal_direction > 0
    short_signal = raw_signal_direction < 0

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength ---
    strength = direction.astype(float) * vol_scalar.fillna(0) * confidence

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
    config: HdMomRevConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.hd_mom_rev.config import ShortMode

    if config.short_mode == ShortMode.DISABLED:
        raw = np.where(long_signal, 1, 0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        drawdown_val = df["drawdown"].shift(1)
        hedge_active = drawdown_val < config.hedge_threshold
        raw = np.where(
            long_signal,
            1,
            np.where(short_signal & hedge_active, -1, 0),
        )

    else:  # FULL
        raw = np.where(long_signal, 1, np.where(short_signal, -1, 0))

    return pd.Series(raw, index=df.index, dtype=int)
