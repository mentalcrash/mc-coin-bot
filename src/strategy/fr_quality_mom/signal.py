"""FR Quality Momentum 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.

핵심 로직:
- 모멘텀 방향으로 진입하되, FR crowding이 높으면 진입하지 않음.
- |fr_zscore| < crowd_threshold: 모멘텀 품질 양호 → 진입 허용.
- |fr_zscore| >= crowd_threshold: crowding 위험 → 진입 차단.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.fr_quality_mom.config import FrQualityMomConfig


def generate_signals(df: pd.DataFrame, config: FrQualityMomConfig) -> StrategySignals:
    """FR Quality Momentum 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.fr_quality_mom.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    momentum = df["momentum"].shift(1)
    fr_zscore = df["fr_zscore"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Quality Filter: FR crowding 낮을 때만 진입 ---
    quality_ok = fr_zscore.abs() < config.fr_crowd_threshold

    # --- Signal Logic ---
    # positive momentum + quality OK = long
    long_signal = (momentum > 0) & quality_ok
    # negative momentum + quality OK = short
    short_signal = (momentum < 0) & quality_ok

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength: conviction by inverse crowding ---
    # lower |fr_zscore| = higher quality = stronger conviction
    raw_quality: pd.Series = 1.0 - fr_zscore.abs() / config.fr_crowd_threshold  # type: ignore[assignment]
    quality_score = raw_quality.clip(lower=0.0)
    strength = direction.astype(float) * vol_scalar.fillna(0) * quality_score.fillna(0)

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
    config: FrQualityMomConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.fr_quality_mom.config import ShortMode

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
