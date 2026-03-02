"""LR-Channel Multi-Scale Trend 시그널 생성.

Rolling OLS 선형회귀 채널 x 3스케일(20/60/150) breakout consensus 기반 시그널.

Shift(1) Rule: LR channel 값은 shift(1) 적용, close는 shift하지 않음.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.lr_channel_trend.config import LrChannelTrendConfig


def generate_signals(df: pd.DataFrame, config: LrChannelTrendConfig) -> StrategySignals:
    """LR-Channel Multi-Scale Trend 시그널 생성.

    Signal Logic:
        1. 각 스케일에 대해 LR channel breakout sub-signal 계산 (총 3개)
           - close > prev_lr_upper -> +1
           - close < prev_lr_lower -> -1
           - else -> 0
        2. consensus = mean(3개 sub-signals)
        3. |consensus| >= entry_threshold -> direction = sign(consensus)
        4. strength = |consensus| * vol_scalar

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.lr_channel_trend.config import ShortMode

    scales = (config.scale_short, config.scale_mid, config.scale_long)

    # --- Shift(1): 전봉 기준 시그널 ---
    vol_scalar = df["vol_scalar"].shift(1)
    dd = df["drawdown"].shift(1)

    # close는 shift하지 않음: 현재 close vs 전봉 channel 비교
    close: pd.Series = df["close"]  # type: ignore[assignment]
    signal_components: list[pd.Series] = []

    # --- LR channel breakout sub-signals ---
    for s in scales:
        prev_upper = df[f"lr_upper_{s}"].shift(1)
        prev_lower = df[f"lr_lower_{s}"].shift(1)
        signal_i = pd.Series(
            np.where(close > prev_upper, 1.0, np.where(close < prev_lower, -1.0, 0.0)),
            index=df.index,
        )
        signal_components.append(signal_i)

    # --- Consensus: 3-signal 평균 ---
    consensus: pd.Series = pd.concat(signal_components, axis=1).mean(axis=1)  # type: ignore[assignment]

    # --- Direction (entry_threshold 적용 + ShortMode 분기) ---
    dd_series: pd.Series = dd  # type: ignore[assignment]
    direction = _compute_direction(consensus, dd_series, config)

    # --- Strength ---
    abs_consensus: pd.Series = consensus.abs()  # type: ignore[assignment]
    strength = direction.astype(float) * abs_consensus * vol_scalar.fillna(0)

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
    consensus: pd.Series,
    drawdown_series: pd.Series,
    config: LrChannelTrendConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.lr_channel_trend.config import ShortMode

    abs_consensus = consensus.abs()
    above_threshold = abs_consensus >= config.entry_threshold

    long_signal = (consensus > 0) & above_threshold
    short_signal = (consensus < 0) & above_threshold

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

    return pd.Series(raw, index=consensus.index, dtype=int)
