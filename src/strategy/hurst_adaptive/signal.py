"""Hurst-Adaptive Signal Generation.

Regime detection:
    - Trending: hurst > 0.55 AND er > 0.40 → momentum following
    - Mean-reverting: hurst < 0.45 AND er < 0.40 → mean reversion
    - Dead zone: no signal (noise reduction)
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.models.types import Direction
from src.strategy.hurst_adaptive.config import HurstAdaptiveConfig
from src.strategy.tsmom.config import ShortMode
from src.strategy.types import StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: HurstAdaptiveConfig | None = None,
) -> StrategySignals:
    """레짐 기반 적응적 시그널 생성.

    Args:
        df: preprocess() 결과 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    if config is None:
        config = HurstAdaptiveConfig()

    required = {"hurst", "er", "trend_momentum", "mr_score", "vol_scalar"}
    missing = required - set(df.columns)
    if missing:
        msg = f"Missing columns: {missing}"
        raise ValueError(msg)

    hurst: pd.Series = df["hurst"]  # type: ignore[assignment]
    er: pd.Series = df["er"]  # type: ignore[assignment]
    trend_mom: pd.Series = df["trend_momentum"]  # type: ignore[assignment]
    mr_score: pd.Series = df["mr_score"]  # type: ignore[assignment]
    vol_scalar: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    # Regime detection
    is_trending = (hurst > config.hurst_trend_threshold) & (er > config.er_trend_threshold)
    is_mr = (hurst < config.hurst_mr_threshold) & (er < config.er_trend_threshold)

    # Direction per regime
    trend_signal = np.sign(trend_mom)
    mr_signal = np.sign(mr_score)

    # Combine: trending → momentum, MR → mean reversion, else → 0
    raw = pd.Series(
        np.where(is_trending, trend_signal, np.where(is_mr, mr_signal, 0.0)),
        index=df.index,
    )

    # Strength: direction * vol_scalar
    raw_strength = raw * vol_scalar

    # Shift(1) — lookahead bias 방지
    signal_shifted = raw_strength.shift(1).fillna(0.0)

    direction = pd.Series(
        np.sign(signal_shifted).fillna(0).astype(int),
        index=df.index,
        name="direction",
    )
    strength = pd.Series(signal_shifted, index=df.index, name="strength")

    # ShortMode 처리
    if config.short_mode == ShortMode.DISABLED:
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)
    elif config.short_mode == ShortMode.HEDGE_ONLY:
        hedge_active = df["drawdown"] < config.hedge_threshold
        short_mask = direction == Direction.SHORT
        suppress_short = short_mask & ~hedge_active
        direction = direction.where(~suppress_short, Direction.NEUTRAL)
        strength = strength.where(~suppress_short, 0.0)

    # Entry / Exit
    prev_dir = direction.shift(1).fillna(0)
    long_entry = (direction == Direction.LONG) & (prev_dir != Direction.LONG)
    short_entry = (direction == Direction.SHORT) & (prev_dir != Direction.SHORT)
    entries = pd.Series(long_entry | short_entry, index=df.index, name="entries")

    to_neutral = (direction == Direction.NEUTRAL) & (prev_dir != Direction.NEUTRAL)
    reversal = direction * prev_dir < 0
    exits = pd.Series(to_neutral | reversal, index=df.index, name="exits")

    # Stats
    n_total = len(df)
    n_long = int((direction == Direction.LONG).sum())
    n_short = int((direction == Direction.SHORT).sum())
    n_trending = int(is_trending.sum())
    n_mr = int(is_mr.sum())
    logger.info(
        "Hurst-Adaptive signals | Total: {}, Long: {} ({:.1%}), Short: {} ({:.1%}), Trending: {}, MR: {}",
        n_total,
        n_long,
        n_long / max(n_total, 1),
        n_short,
        n_short / max(n_total, 1),
        n_trending,
        n_mr,
    )

    return StrategySignals(entries=entries, exits=exits, direction=direction, strength=strength)
