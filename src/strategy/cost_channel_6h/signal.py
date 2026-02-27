"""Cost-Penalized Multi-Scale Channel 시그널 생성.

3종 채널(Donchian/Keltner/BB) x 3스케일(15/45/120) breakout consensus + 비용 페널티.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
Cost Penalty: 포지션 변경 시 기대 이익이 비용을 초과할 때만 시그널 전환.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.cost_channel_6h.config import CostChannel6hConfig


def generate_signals(df: pd.DataFrame, config: CostChannel6hConfig) -> StrategySignals:
    """Cost-Penalized Multi-Scale Channel 시그널 생성.

    Signal Logic:
        1. 각 (채널, 스케일) 조합에 대해 breakout sub-signal 계산 (총 9개)
           - Donchian: close > prev_upper -> +1, close < prev_lower -> -1
           - Keltner: close > prev_upper -> +1, close < prev_lower -> -1
           - BB: close > prev_upper -> +1, close < prev_lower -> -1
        2. consensus = mean(9개 sub-signals)
        3. raw_direction = sign(consensus) if |consensus| >= entry_threshold else 0
        4. cost penalty: 포지션 변경 시 expected_profit > theta * cost 일 때만 변경
        5. strength = |consensus| * vol_scalar

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.cost_channel_6h.config import ShortMode

    scales = (config.scale_short, config.scale_mid, config.scale_long)

    # --- Shift(1): 전봉 기준 시그널 ---
    vol_scalar = df["vol_scalar"].shift(1)
    dd = df["drawdown"].shift(1)
    atr_profit = df["atr_profit"].shift(1)

    # close는 shift하지 않음: 현재 close vs 전봉 channel 비교
    close: pd.Series = df["close"]  # type: ignore[assignment]
    signal_components: list[pd.Series] = []

    # --- Donchian breakout sub-signals ---
    for s in scales:
        prev_upper = df[f"dc_upper_{s}"].shift(1)
        prev_lower = df[f"dc_lower_{s}"].shift(1)
        signal_i = pd.Series(
            np.where(close > prev_upper, 1.0, np.where(close < prev_lower, -1.0, 0.0)),
            index=df.index,
        )
        signal_components.append(signal_i)

    # --- Keltner breakout sub-signals ---
    for s in scales:
        prev_upper = df[f"kc_upper_{s}"].shift(1)
        prev_lower = df[f"kc_lower_{s}"].shift(1)
        signal_i = pd.Series(
            np.where(close > prev_upper, 1.0, np.where(close < prev_lower, -1.0, 0.0)),
            index=df.index,
        )
        signal_components.append(signal_i)

    # --- Bollinger Bands breakout sub-signals ---
    for s in scales:
        prev_upper = df[f"bb_upper_{s}"].shift(1)
        prev_lower = df[f"bb_lower_{s}"].shift(1)
        signal_i = pd.Series(
            np.where(close > prev_upper, 1.0, np.where(close < prev_lower, -1.0, 0.0)),
            index=df.index,
        )
        signal_components.append(signal_i)

    # --- Consensus: 9-signal 평균 ---
    consensus: pd.Series = pd.concat(signal_components, axis=1).mean(axis=1)  # type: ignore[assignment]

    # --- Raw Direction (entry_threshold 적용 + ShortMode 분기) ---
    dd_series: pd.Series = dd  # type: ignore[assignment]
    raw_direction = _compute_direction(consensus, dd_series, config)

    # --- Cost-Penalized Direction ---
    prev_direction = raw_direction.shift(1).fillna(0).astype(int)
    position_change = (raw_direction != prev_direction).astype(float)
    expected_profit = consensus.abs() * atr_profit.fillna(0)
    cost = position_change * config.round_trip_cost

    # Cost filter: 기대 이익 < theta * 비용이면 이전 방향 유지
    cost_exceeded = (expected_profit < config.cost_penalty_theta * cost) & (position_change > 0)
    direction = pd.Series(
        np.where(cost_exceeded, prev_direction, raw_direction),
        index=df.index,
        dtype=int,
    )

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
    config: CostChannel6hConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.cost_channel_6h.config import ShortMode

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
