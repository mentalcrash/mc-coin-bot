"""EMA Multi-Cross 시그널 생성 — 3쌍 합의 투표.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.ema_multi_cross.config import EmaMultiCrossConfig


def generate_signals(df: pd.DataFrame, config: EmaMultiCrossConfig) -> StrategySignals:
    """3쌍 EMA 크로스의 합의 투표로 시그널을 생성한다.

    각 pair의 방향(+1/0/-1)을 합산하여 consensus 점수를 계산.
    min_votes 이상 합의 시에만 포지션 진입.

    Args:
        df: preprocess()로 지표가 추가된 DataFrame.
        config: 전략 설정.

    Returns:
        StrategySignals.
    """
    from src.strategy.ema_multi_cross.config import ShortMode

    # --- Shift(1): 전봉 기준 ---
    cross1 = df["cross1"].shift(1)
    cross2 = df["cross2"].shift(1)
    cross3 = df["cross3"].shift(1)

    vote1 = pd.Series(np.sign(cross1), index=df.index)
    vote2 = pd.Series(np.sign(cross2), index=df.index)
    vote3 = pd.Series(np.sign(cross3), index=df.index)

    consensus: pd.Series = vote1 + vote2 + vote3  # -3 ~ +3

    vol_scalar = df["vol_scalar"].shift(1)

    long_signal: pd.Series = consensus >= config.min_votes
    short_signal: pd.Series = consensus <= -config.min_votes

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength: consensus 강도를 반영 (3/3 합의 > 2/3 합의) ---
    consensus_weight: pd.Series = consensus.abs() / 3.0
    strength = direction.astype(float) * vol_scalar.fillna(0) * consensus_weight.fillna(0)

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
    config: EmaMultiCrossConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.ema_multi_cross.config import ShortMode

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
