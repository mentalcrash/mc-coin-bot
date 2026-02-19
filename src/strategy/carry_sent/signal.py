"""Carry-Sentiment Gate 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.

Signal Logic:
    1. F&G gate (중립 구간): carry 포지션 활성화
       - fg_gate_low <= fg <= fg_gate_high → carry mode
       - carry direction = -sign(avg_FR) (positive FR → short, negative → long)
    2. F&G extreme (contrarian override):
       - fg < fear_threshold → force long (contrarian)
       - fg > greed_threshold → force short (contrarian)
    3. 그 외: neutral (no position)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.carry_sent.config import CarrySentConfig


def generate_signals(df: pd.DataFrame, config: CarrySentConfig) -> StrategySignals:
    """Carry-Sentiment Gate 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.carry_sent.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    avg_fr = df["avg_funding_rate"].shift(1)
    fr_zscore = df["fr_zscore"].shift(1)
    fg = df["oc_fear_greed"].shift(1)
    fg_ma = df["fg_ma"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Zone Classification ---
    # Fear extreme: contrarian long
    fear_extreme = fg < config.fg_fear_threshold

    # Greed extreme: contrarian short
    greed_extreme = fg > config.fg_greed_threshold

    # Neutral gate: carry active zone
    carry_gate = (fg >= config.fg_gate_low) & (fg <= config.fg_gate_high)

    # FR above entry threshold
    fr_active = avg_fr.abs() > config.fr_entry_threshold

    # --- Carry Direction ---
    # -sign(avg_FR): positive FR → short (receive carry), negative FR → long
    carry_dir = pd.Series(-np.sign(avg_fr), index=df.index)

    # --- Combined Signal ---
    # Priority: fear_extreme > greed_extreme > carry_gate
    long_signal = fear_extreme | (carry_gate & fr_active & (carry_dir > 0) & ~greed_extreme)
    short_signal = greed_extreme | (carry_gate & fr_active & (carry_dir < 0) & ~fear_extreme)
    # Mutual exclusion: long takes priority in edge case
    short_signal = short_signal & ~long_signal

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength ---
    # Conviction: FR z-score magnitude (clamped 0~1, from 0~3 range)
    fr_conviction = fr_zscore.abs().clip(upper=3.0).fillna(0) / 3.0

    # F&G deviation from MA (larger deviation = stronger contrarian signal)
    fg_deviation = (fg - fg_ma).abs().fillna(0) / 100.0
    fg_conviction = fg_deviation.clip(upper=1.0)

    # Blended conviction: carry zone uses FR, extremes use F&G
    conviction = pd.Series(
        np.where(fear_extreme | greed_extreme, fg_conviction, fr_conviction),
        index=df.index,
    )
    # Ensure minimum conviction of 0.3 when active
    conviction = conviction.clip(lower=0.3)

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
    config: CarrySentConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.carry_sent.config import ShortMode

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
