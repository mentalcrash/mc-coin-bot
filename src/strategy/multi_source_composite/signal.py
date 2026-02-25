"""Multi-Source Directional Composite 시그널 생성.

3개 직교 소스의 directional sub-signal을 majority vote로 결합.
Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.

Voting Logic:
    - 3/3 합의: direction = 합의 방향, conviction = 1.0
    - 2/3 합의: direction = 합의 방향, conviction = 0.67
    - 1/3 이하: direction = 0 (no position)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.multi_source_composite.config import MultiSourceCompositeConfig

# Conviction levels for vote counts
_CONVICTION_UNANIMOUS = 1.0  # 3/3 agreement
_CONVICTION_MAJORITY = 2.0 / 3.0  # 2/3 agreement


def generate_signals(df: pd.DataFrame, config: MultiSourceCompositeConfig) -> StrategySignals:
    """Multi-Source Directional Composite 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.multi_source_composite.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    mom_dir = df["mom_direction"].shift(1).fillna(0)
    vel_dir = df["velocity_direction"].shift(1).fillna(0)
    fg_dir = df["fg_direction"].shift(1).fillna(0)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Majority Vote ---
    votes_long = (mom_dir > 0).astype(int) + (vel_dir > 0).astype(int) + (fg_dir > 0).astype(int)
    votes_short = (mom_dir < 0).astype(int) + (vel_dir < 0).astype(int) + (fg_dir < 0).astype(int)

    # Direction and conviction from vote count
    # 2+ votes in one direction = signal; 3 votes = higher conviction
    vote_long = votes_long >= 2  # noqa: PLR2004
    vote_short = votes_short >= 2  # noqa: PLR2004

    conviction = pd.Series(0.0, index=df.index)
    conviction = conviction.where(
        ~vote_long,
        np.where(votes_long == 3, _CONVICTION_UNANIMOUS, _CONVICTION_MAJORITY),  # noqa: PLR2004
    )
    conviction = conviction.where(
        ~vote_short,
        np.where(votes_short == 3, _CONVICTION_UNANIMOUS, _CONVICTION_MAJORITY),  # noqa: PLR2004
    )

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        vote_long=vote_long,
        vote_short=vote_short,
        df=df,
        config=config,
    )

    # --- Strength ---
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
    vote_long: pd.Series,
    vote_short: pd.Series,
    df: pd.DataFrame,
    config: MultiSourceCompositeConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.multi_source_composite.config import ShortMode

    if config.short_mode == ShortMode.DISABLED:
        raw = np.where(vote_long, 1, 0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        dd = df["drawdown"].shift(1)
        hedge_active = dd < config.hedge_threshold
        raw = np.where(
            vote_long,
            1,
            np.where(vote_short & hedge_active, -1, 0),
        )

    else:  # FULL
        raw = np.where(vote_long, 1, np.where(vote_short, -1, 0))

    return pd.Series(raw, index=df.index, dtype=int)
