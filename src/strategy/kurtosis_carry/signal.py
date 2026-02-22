"""Kurtosis Carry 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.kurtosis_carry.config import KurtosisCarryConfig


def generate_signals(df: pd.DataFrame, config: KurtosisCarryConfig) -> StrategySignals:
    """Kurtosis Carry 시그널 생성.

    Long: kurtosis z-score가 하락 (정상화) + momentum 상승
          → fat tail 리스크 프리미엄 수취 구간.
    Short: kurtosis z-score 급등 (극단 tail risk) + momentum 하락
           → tail risk 회피.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.kurtosis_carry.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    kurt_zscore = df["kurtosis_zscore"].shift(1)
    momentum = df["momentum"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    # Long: kurtosis normalizing (z-score low/falling) + positive momentum
    # → tail risk premium 수취 구간: 높았던 kurtosis가 정상화되며 프리미엄 축적
    long_signal = (kurt_zscore < config.low_kurtosis_zscore) & (momentum > 0)

    # Short: kurtosis extreme high (fat tails active) + negative momentum
    # → tail risk 활성화 구간: 급락 가능성
    short_signal = (kurt_zscore > config.high_kurtosis_zscore) & (momentum < 0)

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
    config: KurtosisCarryConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.kurtosis_carry.config import ShortMode

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
