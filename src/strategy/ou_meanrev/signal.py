"""OU Mean Reversion Signal Generator.

OU process half-life 기반 mean reversion 시그널 생성.

Signal Formula:
    1. Shift(1) on ou_zscore, half_life, vol_scalar
    2. MR active = half_life_prev < max_half_life (fast reversion regime)
    3. Long: ou_zscore_prev < -entry_zscore AND mr_active
    4. Short: ou_zscore_prev > +entry_zscore AND mr_active
    5. strength = direction * vol_scalar
    6. Exit: |zscore| < exit_zscore OR half_life > max_half_life OR timeout

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.ou_meanrev.config import OUMeanRevConfig, ShortMode
from src.strategy.types import Direction, StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: OUMeanRevConfig | None = None,
) -> StrategySignals:
    """OU Mean Reversion 시그널 생성.

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
        config: OU Mean Reversion 설정. None이면 기본 설정 사용.

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    if config is None:
        config = OUMeanRevConfig()

    required_cols = {"ou_zscore", "half_life", "vol_scalar"}
    if config.short_mode == ShortMode.HEDGE_ONLY:
        required_cols.add("drawdown")

    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 1. Shift(1) 적용: 전봉 기준 시그널
    zscore_prev: pd.Series = df["ou_zscore"].shift(1)  # type: ignore[assignment]
    half_life_prev: pd.Series = df["half_life"].shift(1)  # type: ignore[assignment]
    vol_scalar_prev: pd.Series = df["vol_scalar"].shift(1)  # type: ignore[assignment]

    # 2. MR active = half_life < max_half_life (positive theta, fast mean reversion)
    mr_active = half_life_prev < config.max_half_life

    # 3. Entry conditions (mean reversion: buy low, sell high)
    long_cond = (zscore_prev < -config.entry_zscore) & mr_active
    short_cond = (zscore_prev > config.entry_zscore) & mr_active

    # 4. Exit conditions
    exit_zscore_cond = zscore_prev.abs() < config.exit_zscore
    exit_halflife_cond = half_life_prev >= config.max_half_life

    # 5. Vectorized position tracking (ffill + cumsum + groupby pattern)
    # Entry → direction, exit → 0, hold → NaN (ffill maintains position)
    raw_direction = pd.Series(
        np.where(
            exit_zscore_cond | exit_halflife_cond,
            0,
            np.where(long_cond, 1, np.where(short_cond, -1, np.nan)),
        ),
        index=df.index,
    )
    position = raw_direction.ffill().fillna(0).astype(int)

    # 6. Timeout: count consecutive bars in same direction
    direction_change = position != position.shift(1)
    direction_group = direction_change.cumsum()
    bars_in_position = direction_group.groupby(direction_group).cumcount()
    timed_out = (bars_in_position >= config.exit_timeout_bars) & (position != 0)
    position = position.where(~timed_out, 0)

    # 7. Strength = position * vol_scalar
    strength_raw = position * vol_scalar_prev.fillna(0)

    # 8. Direction normalization
    direction = pd.Series(
        np.sign(strength_raw).fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    strength = pd.Series(
        strength_raw.fillna(0),
        index=df.index,
        name="strength",
    )

    # 9. 숏 모드에 따른 시그널 처리
    if config.short_mode == ShortMode.DISABLED:
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        drawdown_series: pd.Series = df["drawdown"].shift(1)  # type: ignore[assignment]
        hedge_active = drawdown_series < config.hedge_threshold

        short_mask = direction == Direction.SHORT
        suppress_short = short_mask & ~hedge_active
        direction = direction.where(~suppress_short, Direction.NEUTRAL)
        strength = strength.where(~suppress_short, 0.0)

        active_short = short_mask & hedge_active
        strength = strength.where(
            ~active_short,
            strength * config.hedge_strength_ratio,
        )

        hedge_days = int(hedge_active.sum())
        if hedge_days > 0:
            logger.info(
                "Hedge Mode | Active: {} days ({:.1f}%), Threshold: {:.1f}%",
                hedge_days,
                hedge_days / len(hedge_active) * 100,
                config.hedge_threshold * 100,
            )

    # 10. 진입/청산 시그널
    prev_direction = direction.shift(1).fillna(0)

    long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)
    short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)

    entries = pd.Series(
        long_entry | short_entry,
        index=df.index,
        name="entries",
    )

    to_neutral = (direction == Direction.NEUTRAL) & (prev_direction != Direction.NEUTRAL)
    reversal = direction * prev_direction < 0

    exits = pd.Series(
        to_neutral | reversal,
        index=df.index,
        name="exits",
    )

    # 디버그: 시그널 통계
    valid_strength = strength[strength != 0]
    long_signals = strength[strength > 0]
    short_signals = strength[strength < 0]

    if len(valid_strength) > 0:
        logger.info(
            "Signal Statistics | Total: {} signals, Long: {} ({:.1f}%), Short: {} ({:.1f}%)",
            len(valid_strength),
            len(long_signals),
            len(long_signals) / len(valid_strength) * 100,
            len(short_signals),
            len(short_signals) / len(valid_strength) * 100,
        )

    return StrategySignals(
        entries=entries,
        exits=exits,
        direction=direction,
        strength=strength,
    )
