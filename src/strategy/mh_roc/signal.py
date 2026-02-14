"""Multi-Horizon ROC Ensemble 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
4개 horizon ROC의 부호 투표로 방향 결정. vote_threshold 이상 일치 시 진입.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.mh_roc.config import MhRocConfig


def generate_signals(df: pd.DataFrame, config: MhRocConfig) -> StrategySignals:
    """Multi-Horizon ROC Ensemble 시그널 생성.

    Signal Logic:
        - vote_sum >= vote_threshold → Long (다수 horizon 상승)
        - vote_sum <= -vote_threshold → Short (다수 horizon 하락)
        - |vote_sum| < vote_threshold → Neutral
        - strength = direction * vol_scalar * |vote_ratio| (conviction weighting)

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.mh_roc.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    vote_sum = df["vote_sum"].shift(1)
    vote_ratio = df["vote_ratio"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    long_signal = vote_sum >= config.vote_threshold
    short_signal = vote_sum <= -config.vote_threshold

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength (conviction-weighted) ---
    # |vote_ratio|를 conviction으로 사용: 만장일치(1.0) vs 3:1(0.5)
    conviction = vote_ratio.abs().fillna(0).clip(upper=1.0)
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
    config: MhRocConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.mh_roc.config import ShortMode

    if config.short_mode == ShortMode.DISABLED:
        raw = np.where(long_signal, 1, 0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        drawdown = df["drawdown"].shift(1)
        hedge_active = drawdown < config.hedge_threshold
        raw = np.where(
            long_signal,
            1,
            np.where(short_signal & hedge_active, -1, 0),
        )

    else:  # FULL
        raw = np.where(long_signal, 1, np.where(short_signal, -1, 0))

    return pd.Series(raw, index=df.index, dtype=int)
