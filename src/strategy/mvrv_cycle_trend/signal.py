"""MVRV Cycle Trend 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.

MVRV Z-Score 사이클 레짐 필터:
    - mvrv_zscore < bull_threshold → 저평가 구간 (bull regime) → long 강화
    - mvrv_zscore > bear_threshold → 과대평가 구간 (bear regime) → short 강화
    - 그 외 → 중립 레짐 → 순수 momentum

Momentum:
    - mom_blend > 0 → long signal
    - mom_blend < 0 → short signal

Direction 결합:
    - bull regime + positive momentum → long
    - bear regime + negative momentum → short
    - 중립 regime → momentum 방향 그대로 (conviction 감소)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.mvrv_cycle_trend.config import MvrvCycleTrendConfig


def generate_signals(df: pd.DataFrame, config: MvrvCycleTrendConfig) -> StrategySignals:
    """MVRV Cycle Trend 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.mvrv_cycle_trend.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    mom_blend: pd.Series = df["mom_blend"].shift(1)  # type: ignore[assignment]
    mvrv_z: pd.Series = df["mvrv_zscore"].shift(1)  # type: ignore[assignment]
    vol_scalar: pd.Series = df["vol_scalar"].shift(1)  # type: ignore[assignment]
    dd: pd.Series = df["drawdown"].shift(1)  # type: ignore[assignment]

    # --- MVRV Cycle Regime ---
    # bull regime: MVRV Z < bull_threshold (저평가 → 상승 사이클)
    # bear regime: MVRV Z > bear_threshold (과대평가 → 하락 사이클)
    # neutral: 그 외
    # NaN (on-chain 부재) → 중립 레짐 (Graceful Degradation)
    is_bull_regime = mvrv_z < config.mvrv_bull_threshold
    is_bear_regime = mvrv_z > config.mvrv_bear_threshold
    has_mvrv = mvrv_z.notna()

    # --- Momentum Signal ---
    mom_long = mom_blend > 0
    mom_short = mom_blend < 0

    # --- Direction 결합 (regime filter + momentum) ---
    # Bull regime: long만 허용 (momentum long → long, momentum short → neutral)
    # Bear regime: short만 허용 (momentum short → short, momentum long → neutral)
    # Neutral/NaN regime: momentum 방향 그대로
    long_signal = (is_bull_regime & mom_long) | (~has_mvrv & mom_long)
    short_signal = (is_bear_regime & mom_short) | (~has_mvrv & mom_short)

    # Neutral regime (MVRV between thresholds): momentum 방향 허용
    neutral_regime = has_mvrv & ~is_bull_regime & ~is_bear_regime
    long_signal = long_signal | (neutral_regime & mom_long)
    short_signal = short_signal | (neutral_regime & mom_short)

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        drawdown_series=dd,
        config=config,
    )

    # --- Conviction (MVRV regime alignment bonus) ---
    # Regime-aligned: conviction 1.0, regime-neutral: conviction 0.7
    conviction = pd.Series(0.7, index=df.index)
    conviction = conviction.where(
        ~(is_bull_regime & (direction == 1)),
        1.0,
    )
    conviction = conviction.where(
        ~(is_bear_regime & (direction == -1)),
        1.0,
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
    long_signal: pd.Series,
    short_signal: pd.Series,
    drawdown_series: pd.Series,
    config: MvrvCycleTrendConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.mvrv_cycle_trend.config import ShortMode

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

    return pd.Series(raw, index=long_signal.index, dtype=int)
