"""Complexity-Filtered Trend 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.

복잡도가 낮은 구간(Hurst > threshold + Fractal < threshold + ER > threshold)에서만
추세추종 시그널 활성화. 높은 복잡도에서는 관망.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.complexity_trend.config import ComplexityTrendConfig

# 3개 복잡도 지표 중 최소 충족 개수 (Hurst, Fractal, ER)
_MIN_COMPLEXITY_PASS = 2
_TOTAL_COMPLEXITY_INDICATORS = 3


def generate_signals(df: pd.DataFrame, config: ComplexityTrendConfig) -> StrategySignals:
    """Complexity-Filtered Trend 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.complexity_trend.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    hurst = df["hurst"].shift(1)
    fractal_dim = df["fractal_dim"].shift(1)
    er = df["efficiency"].shift(1)
    close_prev = df["close"].shift(1)
    trend_sma = df["trend_sma"].shift(1)
    trend_roc = df["trend_roc"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Complexity Filter ---
    # 3가지 복잡도 지표 중 2개 이상 만족 시 "예측 가능한 시장"으로 판단
    cond_hurst = hurst > config.hurst_threshold
    cond_fractal = fractal_dim < config.fractal_threshold
    cond_er = er > config.er_threshold

    complexity_score = cond_hurst.astype(int) + cond_fractal.astype(int) + cond_er.astype(int)
    low_complexity = complexity_score >= _MIN_COMPLEXITY_PASS

    # --- Trend Direction ---
    uptrend = close_prev > trend_sma
    downtrend = close_prev < trend_sma

    # --- Signal: 낮은 복잡도 + 추세 방향 ---
    long_signal = low_complexity & uptrend & (trend_roc > 0)
    short_signal = low_complexity & downtrend & (trend_roc < 0)

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength: direction * vol_scalar * complexity conviction ---
    # complexity_score를 0~1 conviction으로 변환 (2/3 or 3/3)
    conviction = (
        complexity_score.clip(0, _TOTAL_COMPLEXITY_INDICATORS).astype(float)
        / _TOTAL_COMPLEXITY_INDICATORS
    )
    strength = direction.astype(float) * vol_scalar.fillna(0) * conviction.fillna(0)

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
    config: ComplexityTrendConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.complexity_trend.config import ShortMode

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
