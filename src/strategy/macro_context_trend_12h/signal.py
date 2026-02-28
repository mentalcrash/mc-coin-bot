"""Macro-Context-Trend 12H signal generator.

EMA cross 추세 확인 + 매크로 컨텍스트 사이징.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.macro_context_trend_12h.config import ShortMode

if TYPE_CHECKING:
    from src.strategy.macro_context_trend_12h.config import MacroContextTrendConfig
    from src.strategy.types import StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: MacroContextTrendConfig,
) -> StrategySignals:
    """시그널 생성 (shift(1) 적용).

    Args:
        df: preprocessed DataFrame.
        config: 전략 설정.

    Returns:
        StrategySignals (entries, exits, direction, strength).
    """
    from src.strategy.types import StrategySignals

    # --- Shift(1) 적용: 이전 bar 데이터 사용 ---
    vol_scalar = df["vol_scalar"].shift(1)
    trend_dir = df["trend_direction"].shift(1)
    trend_confirmed = df["trend_confirmed"].shift(1)
    macro_ctx = df["macro_context"].shift(1)

    # --- 확인된 추세 시그널 ---
    long_signal = (trend_dir > 0) & (trend_confirmed > 0)
    short_signal = (trend_dir < 0) & (trend_confirmed > 0)

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(long_signal, short_signal, df, config)

    # --- Strength: vol_scalar x macro_context ---
    strength = direction.astype(float) * vol_scalar.fillna(0) * macro_ctx.fillna(1.0)

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
    config: MacroContextTrendConfig,
) -> pd.Series:
    """ShortMode 분기 처리.

    Args:
        long_signal: Long 진입 조건 Series.
        short_signal: Short 진입 조건 Series.
        df: preprocessed DataFrame.
        config: 전략 설정.

    Returns:
        Direction Series (-1, 0, 1).
    """
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
