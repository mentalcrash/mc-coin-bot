"""Persistence-Weighted-Trend 12H signal generator.

3-scale persistence score 앙상블 x ROC 방향 -> 추세 시그널.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.persistence_weighted_trend_12h.config import ShortMode

if TYPE_CHECKING:
    from src.strategy.persistence_weighted_trend_12h.config import PersistenceWeightedTrendConfig
    from src.strategy.types import StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: PersistenceWeightedTrendConfig,
) -> StrategySignals:
    """시그널 생성 (shift(1) 적용).

    Args:
        df: preprocessed DataFrame.
        config: 전략 설정.

    Returns:
        StrategySignals (entries, exits, direction, strength).
    """
    from src.strategy.types import StrategySignals

    scales = (config.scale_short, config.scale_mid, config.scale_long)

    # --- Shift(1) 적용: 이전 bar 데이터 사용 ---
    vol_scalar = df["vol_scalar"].shift(1)
    roc_dir = df["roc_direction"].shift(1)

    # --- Multi-scale persistence 앙상블 ---
    persistence_components: list[pd.Series] = []
    for s in scales:
        p: pd.Series = df[f"persistence_{s}"].shift(1)  # type: ignore[assignment]
        persistence_components.append(p)

    # 3-scale 평균 persistence score
    avg_persistence: pd.Series = pd.concat(persistence_components, axis=1).mean(axis=1)  # type: ignore[assignment]

    # --- Persistence 임계값 초과 + 모멘텀 방향 ---
    persistence_active = avg_persistence >= config.persistence_threshold
    long_signal: pd.Series = persistence_active & (roc_dir > 0)  # type: ignore[assignment]
    short_signal: pd.Series = persistence_active & (roc_dir < 0)  # type: ignore[assignment]

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(long_signal, short_signal, df, config)

    # --- Strength: persistence x vol_scalar ---
    strength = direction.astype(float) * avg_persistence.fillna(0) * vol_scalar.fillna(0)  # type: ignore[union-attr]

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
    config: PersistenceWeightedTrendConfig,
) -> pd.Series:
    """ShortMode 분기 처리.

    Args:
        long_signal: Long 진입 조건 Series.
        short_signal: Short 진입 조건 Series.
        df: preprocessed DataFrame.
        config: 전략 설정.

    Returns:
        Direction Series (-1, 0, 1).
    """
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
