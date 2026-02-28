"""Trend Factor Multi-Horizon 시그널 생성.

5-horizon risk-adjusted return 합산으로 multi-scale momentum consensus 판단.
tanh 스케일링으로 연속적 포지션 사이징.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.trend_factor_12h.config import TrendFactorConfig


def generate_signals(df: pd.DataFrame, config: TrendFactorConfig) -> StrategySignals:
    """Trend Factor Multi-Horizon 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.trend_factor_12h.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    trend_factor = df["trend_factor"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    abs_tf = trend_factor.abs()
    above_threshold = abs_tf >= config.entry_threshold

    long_signal = (trend_factor > 0) & above_threshold
    short_signal = (trend_factor < 0) & above_threshold

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength: direction * tanh(|trend_factor| * scale) * vol_scalar ---
    conviction = np.tanh(abs_tf.fillna(0) * config.tanh_scale)
    strength = direction.astype(float) * conviction * vol_scalar.fillna(0)

    if config.short_mode == ShortMode.HEDGE_ONLY:
        strength = pd.Series(
            np.where(
                direction == -1,
                strength * config.hedge_strength_ratio,
                strength,
            ),
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
    config: TrendFactorConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.trend_factor_12h.config import ShortMode

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
