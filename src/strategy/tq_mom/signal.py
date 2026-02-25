"""Trend Quality Momentum (TQ-Mom) 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.

Signal Logic:
    - Hurst > threshold (추세 지속) + FD < threshold (정돈) → quality pass
    - quality pass + price_mom 방향 → 진입
    - conviction = (hurst - 0.5).clip(0, 0.5) * (2 - fd).clip(0, 1)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.tq_mom.config import TqMomConfig


def generate_signals(df: pd.DataFrame, config: TqMomConfig) -> StrategySignals:
    """Trend Quality Momentum 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.tq_mom.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    hurst = df["hurst"].shift(1)
    fd = df["fd"].shift(1)
    price_mom = df["price_mom"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    # Quality gate: Hurst > threshold (추세 지속) AND FD < threshold (정돈)
    quality_pass = (hurst > config.hurst_threshold) & (fd < config.fd_threshold)

    long_signal = quality_pass & (price_mom > 0)
    short_signal = quality_pass & (price_mom < 0)

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Conviction ---
    # (hurst - 0.5): 0.5 초과분이 추세 신뢰도 (0~0.5)
    # (2 - fd): FD가 1에 가까울수록 높음, 2에 가까울수록 낮음 (0~1)
    hurst_excess = (hurst.fillna(0.5) - 0.5).clip(lower=0.0, upper=0.5)
    fd_quality = (2.0 - fd.fillna(1.5)).clip(lower=0.0, upper=1.0)
    # Normalize to 0~1 range: max(hurst_excess)=0.5, max(fd_quality)=1.0
    conviction = (hurst_excess * 2.0) * fd_quality  # scale hurst_excess to 0~1

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
    config: TqMomConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.tq_mom.config import ShortMode

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
