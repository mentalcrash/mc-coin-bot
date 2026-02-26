"""Donchian Filtered 시그널 생성.

Donch-Multi 3-scale consensus + funding rate crowd filter.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.donch_filtered.config import DonchFilteredConfig


def generate_signals(df: pd.DataFrame, config: DonchFilteredConfig) -> StrategySignals:
    """Donchian Filtered 시그널 생성.

    Signal Logic:
        1. Donch-Multi와 동일한 3-scale consensus 계산
        2. Crowd filter: fr_zscore 극단값 + 동일 방향 → 진입 억제
        3. |consensus| >= entry_threshold 시 direction = sign(consensus)
        4. strength = |consensus| * vol_scalar

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.donch_multi.config import ShortMode

    lookbacks = (config.lookback_short, config.lookback_mid, config.lookback_long)

    # --- Shift(1): 전봉 기준 시그널 ---
    vol_scalar = df["vol_scalar"].shift(1)
    dd = df["drawdown"].shift(1)
    fr_zscore: pd.Series = df["fr_zscore"].shift(1).fillna(0.0)  # type: ignore[assignment]

    # --- Per-scale breakout 시그널 ---
    close: pd.Series = df["close"]  # type: ignore[assignment]
    signal_components: list[pd.Series] = []

    for lb in lookbacks:
        prev_upper = df[f"dc_upper_{lb}"].shift(1)
        prev_lower = df[f"dc_lower_{lb}"].shift(1)

        signal_i = pd.Series(
            np.where(
                close > prev_upper,
                1.0,
                np.where(close < prev_lower, -1.0, 0.0),
            ),
            index=df.index,
        )
        signal_components.append(signal_i)

    # --- Consensus: 3-scale 평균 ---
    consensus: pd.Series = pd.concat(signal_components, axis=1).mean(axis=1)  # type: ignore[assignment]

    # --- Direction (entry_threshold 적용 + ShortMode 분기) ---
    dd_series: pd.Series = dd  # type: ignore[assignment]
    direction = _compute_direction(consensus, dd_series, config)

    # --- Crowd Filter: 과열 방향 진입 억제 ---
    direction = _apply_crowd_filter(direction, fr_zscore, config.fr_suppress_threshold)

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
    config: DonchFilteredConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.donch_multi.config import ShortMode

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


def _apply_crowd_filter(
    direction: pd.Series,
    fr_zscore: pd.Series,
    threshold: float,
) -> pd.Series:
    """Crowd filter: 과열 포지셔닝 방향 진입 억제.

    - direction == +1 AND fr_zscore > threshold → 롱 과열, 억제 (→ 0)
    - direction == -1 AND fr_zscore < -threshold → 숏 과열, 억제 (→ 0)
    - 그 외 → 통과

    Args:
        direction: 원시 direction 시리즈.
        fr_zscore: funding rate z-score (shift(1) 적용 완료).
        threshold: 억제 임계값.

    Returns:
        필터링된 direction 시리즈.
    """
    suppress = ((direction == 1) & (fr_zscore > threshold)) | (
        (direction == -1) & (fr_zscore < -threshold)
    )
    filtered = pd.Series(
        np.where(suppress, 0, direction),
        index=direction.index,
        dtype=int,
    )
    return filtered
