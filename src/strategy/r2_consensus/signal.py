"""R2 Consensus Trend 시그널 생성.

3개 스케일(short/mid/long)에서 R2 > threshold인 스케일만 투표 참여.
각 투표 = sign(slope). consensus = mean(votes).
|consensus| >= entry_threshold이면 진입.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.r2_consensus.config import R2ConsensusConfig


def generate_signals(df: pd.DataFrame, config: R2ConsensusConfig) -> StrategySignals:
    """R2 Consensus Trend 시그널 생성.

    Signal Logic:
        1. 각 스케일(short/mid/long)에서 shift(1) 적용된 R^2, slope 로드
        2. R^2 >= r2_threshold이면 vote = sign(slope), 아니면 vote = 0
        3. consensus = mean(votes)  (범위: -1 ~ +1)
        4. |consensus| >= entry_threshold 시 direction = sign(consensus)
        5. strength = direction * |consensus| * vol_scalar

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.r2_consensus.config import ShortMode

    lookbacks = (config.lookback_short, config.lookback_mid, config.lookback_long)

    # --- Shift(1): 전봉 기준 시그널 ---
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Per-scale voting ---
    votes: list[pd.Series] = []
    for lb in lookbacks:
        r2 = df[f"r2_{lb}"].shift(1)
        slope = df[f"slope_{lb}"].shift(1)

        # R^2가 threshold 이상이면 sign(slope)으로 투표, 아니면 기권(0)
        vote = pd.Series(
            np.where(r2 >= config.r2_threshold, np.sign(slope), 0.0),
            index=df.index,
        )
        votes.append(vote)

    # --- Consensus: 3-scale 투표 평균 ---
    consensus: pd.Series = pd.concat(votes, axis=1).mean(axis=1)  # type: ignore[assignment]

    # --- Direction (entry_threshold 적용 + ShortMode 분기) ---
    dd: pd.Series = df["drawdown"].shift(1)  # type: ignore[assignment]
    direction = _compute_direction(consensus, dd, config)

    # --- Strength: direction * |consensus| * vol_scalar ---
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
    config: R2ConsensusConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.r2_consensus.config import ShortMode

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
