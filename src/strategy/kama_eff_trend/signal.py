"""KAMA Efficiency Trend 시그널 생성.

ER >= threshold인 구간에서 KAMA slope 방향으로 진입.
KAMA-price 거리(ATR 정규화)를 conviction으로 사용.
Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.kama_eff_trend.config import ShortMode
from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.kama_eff_trend.config import KamaEffTrendConfig

# Conviction 최대 거리 (ATR 배수). 이 이상은 3.0으로 cap.
_MAX_KAMA_DIST = 3.0


def generate_signals(df: pd.DataFrame, config: KamaEffTrendConfig) -> StrategySignals:
    """KAMA Efficiency Trend 시그널 생성.

    Signal Logic:
        1. ER >= er_threshold -> 추세 품질 필터 통과
        2. kama_slope > 0 -> long, kama_slope < 0 -> short
        3. conviction = min(|kama_dist|, 3.0) / 3.0 (0~1 정규화)
        4. strength = direction * vol_scalar * conviction

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    # --- Shift(1): 전봉 기준 시그널 ---
    er = df["er"].shift(1)
    kama_slope = df["kama_slope"].shift(1)
    kama_dist = df["kama_dist"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    er_pass = er >= config.er_threshold
    long_signal = er_pass & (kama_slope > 0)
    short_signal = er_pass & (kama_slope < 0)

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Conviction: ATR-normalized KAMA-price distance (capped) ---
    conviction = kama_dist.abs().clip(upper=_MAX_KAMA_DIST) / _MAX_KAMA_DIST
    conviction = conviction.fillna(0.0)

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
    long_signal: pd.Series,
    short_signal: pd.Series,
    df: pd.DataFrame,
    config: KamaEffTrendConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
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
