"""Participation Momentum 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
거래 참여도(intensity) Z-score가 높을 때 + 모멘텀 방향 일치 → 진입.
tflow_intensity 부재 시 순수 모멘텀 방향으로 fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.participation_mom_12h.config import ParticipationMomConfig


def generate_signals(df: pd.DataFrame, config: ParticipationMomConfig) -> StrategySignals:
    """Participation Momentum 시그널 생성.

    Signal Logic:
        - intensity_zscore >= intensity_long_z AND mom_direction == 1 → Long
        - intensity_zscore <= intensity_short_z AND mom_direction == -1 → Short
        - tflow_intensity 부재 시(zscore==0): 순수 EMA 모멘텀 방향 fallback

    Strength = direction * vol_scalar.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.participation_mom_12h.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    intensity_z = df["intensity_zscore"].shift(1).fillna(0.0)
    mom_dir = df["mom_direction"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)
    mom_strength = df["mom_strength"].shift(1).fillna(0.0)

    # --- Signal Logic ---
    # Trade Flow 존재 시: intensity Z-score + momentum direction 결합
    has_tflow = (df["intensity_zscore"] != 0).any()

    if has_tflow:
        # 높은 참여도 + 상승 모멘텀 → Long
        long_signal = (intensity_z >= config.intensity_long_z) & (mom_dir == 1)
        # 높은 참여도(절대값) + 하락 모멘텀 → Short
        short_signal = (intensity_z <= config.intensity_short_z) & (mom_dir == -1)
    else:
        # Graceful Degradation: 순수 EMA 모멘텀 방향
        long_signal = mom_dir == 1
        short_signal = mom_dir == -1

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength ---
    # 확신도: momentum strength 크기에 비례 (최대 1.5x)
    conviction = (1.0 + mom_strength.abs().clip(upper=0.5)).fillna(1.0)
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
    config: ParticipationMomConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.participation_mom_12h.config import ShortMode

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
