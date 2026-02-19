"""Liquidity-Confirmed Trend 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.

Signal Logic:
    1. Base: price_mom direction (Long if positive, Short if negative)
    2. Confirmation: liq_score >= threshold (liquidity must confirm)
       - Both stablecoin + TVL growing = strong confirmation
    3. F&G Override (contrarian):
       - fg < fear_threshold → force long (regardless of mom/liq)
       - fg > greed_threshold → force short (regardless of mom/liq)
    4. Without F&G data → pure momentum + liquidity confirmation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.liq_conf_trend.config import LiqConfTrendConfig


def generate_signals(df: pd.DataFrame, config: LiqConfTrendConfig) -> StrategySignals:
    """Liquidity-Confirmed Trend 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.liq_conf_trend.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    price_mom = df["price_mom"].shift(1)
    liq_score = df["liq_score"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)
    fg = df["oc_fear_greed"].shift(1)

    # --- F&G Override (contrarian) ---
    has_fg = fg.notna()
    fear_extreme = has_fg & (fg < config.fg_fear_threshold)
    greed_extreme = has_fg & (fg > config.fg_greed_threshold)

    # --- Base Momentum + Liquidity Confirmation ---
    mom_long = price_mom > 0
    mom_short = price_mom < 0
    liq_confirmed = liq_score >= config.liq_score_threshold

    # --- Combined Signals ---
    # Priority: F&G extreme > momentum + liquidity
    long_signal = fear_extreme | (mom_long & liq_confirmed & ~greed_extreme)
    short_signal = greed_extreme | (mom_short & liq_confirmed & ~fear_extreme)

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength ---
    # Conviction from liquidity score magnitude (0~2 → 0.5~1.0)
    liq_conviction = (liq_score.fillna(0) / 2.0).clip(lower=0.0, upper=1.0)
    # F&G extremes get full conviction
    conviction = pd.Series(
        np.where(fear_extreme | greed_extreme, 1.0, 0.5 + liq_conviction * 0.5),
        index=df.index,
    )

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
    config: LiqConfTrendConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.liq_conf_trend.config import ShortMode

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
