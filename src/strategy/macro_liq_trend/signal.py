"""Macro-Liquidity Adaptive Trend 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.macro_liq_trend.config import MacroLiqTrendConfig


def generate_signals(df: pd.DataFrame, config: MacroLiqTrendConfig) -> StrategySignals:
    """Macro-Liquidity Adaptive Trend 시그널 생성.

    매크로 유동성 composite score와 가격 모멘텀 정렬로
    Long/Short/Neutral 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.macro_liq_trend.config import ShortMode

    close: pd.Series = df["close"]  # type: ignore[assignment]

    # --- Shift(1): 전봉 기준 시그널 ---
    macro_liq_score = df["macro_liq_score"].shift(1)
    sma_price = df["sma_price"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)
    dd: pd.Series = df["drawdown"].shift(1)  # type: ignore[assignment]

    # --- Price Momentum Conditions ---
    price_above_sma = close.shift(1) > sma_price
    price_below_sma = close.shift(1) < sma_price

    # --- Macro Liquidity Conditions ---
    liq_bullish = macro_liq_score > config.liq_long_threshold
    liq_bearish = macro_liq_score < config.liq_short_threshold

    # --- Combined Signals (2 independent: macro + price mom) ---
    long_signal = liq_bullish & price_above_sma
    short_signal = liq_bearish & price_below_sma

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        drawdown_series=dd,
        config=config,
        index=df.index,
    )

    # --- Strength ---
    base_scalar = vol_scalar.fillna(0)
    strength = direction.astype(float) * base_scalar

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
    drawdown_series: pd.Series,
    config: MacroLiqTrendConfig,
    index: pd.Index,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.macro_liq_trend.config import ShortMode

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

    return pd.Series(raw, index=index, dtype=int)
