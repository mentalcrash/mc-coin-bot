"""CVD Divergence 8H 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.

Signal Logic:
    1. Trend direction: close > EMA → uptrend (+1), close < EMA → downtrend (-1)
    2. CVD divergence confirmation:
       - Bullish div (divergence_zscore < -threshold): CVD 상승 > 가격 → 매수 압력
       - Bearish div (divergence_zscore > +threshold): 가격 상승 > CVD → 매도 압력
    3. Combined:
       - Uptrend + no bearish divergence → long
       - Downtrend + no bullish divergence → short (FULL mode)
       - Divergence → signal 반전 또는 중립
    4. Graceful degradation: CVD 없으면 (divergence_zscore=0) pure EMA trend
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.cvd_diverge_8h.config import CvdDiverge8hConfig


def generate_signals(df: pd.DataFrame, config: CvdDiverge8hConfig) -> StrategySignals:
    """CVD Divergence 8H 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.cvd_diverge_8h.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    close_prev = df["close"].shift(1)
    trend_ema_prev = df["trend_ema"].shift(1)
    div_zscore_prev = df["divergence_zscore"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Trend Direction ---
    # Above EMA = uptrend (+1), below = downtrend (-1)
    trend_dir = pd.Series(
        np.where(close_prev > trend_ema_prev, 1, np.where(close_prev < trend_ema_prev, -1, 0)),
        index=df.index,
    )

    # --- CVD Divergence Detection ---
    threshold = config.divergence_threshold
    bearish_div = div_zscore_prev > threshold  # price outpacing CVD
    bullish_div = div_zscore_prev < -threshold  # CVD outpacing price

    # --- Combined Signal ---
    # Uptrend + no bearish divergence → long
    # Uptrend + bearish divergence → neutral (divergence warns of reversal)
    # Downtrend + no bullish divergence → short
    # Downtrend + bullish divergence → neutral (divergence warns of reversal)
    long_signal = (trend_dir == 1) & ~bearish_div
    short_signal = (trend_dir == -1) & ~bullish_div

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength ---
    strength = direction.astype(float) * vol_scalar.fillna(0)

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
    config: CvdDiverge8hConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.cvd_diverge_8h.config import ShortMode

    if config.short_mode == ShortMode.DISABLED:
        raw = np.where(long_signal, 1, 0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        drawdown_prev = df["drawdown"].shift(1)
        hedge_active = drawdown_prev < config.hedge_threshold
        raw = np.where(
            long_signal,
            1,
            np.where(short_signal & hedge_active, -1, 0),
        )

    else:  # FULL
        raw = np.where(long_signal, 1, np.where(short_signal, -1, 0))

    return pd.Series(raw, index=df.index, dtype=int)
