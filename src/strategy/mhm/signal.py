"""MHM 시그널 생성.

다중 horizon 모멘텀 agreement + 역변동성 가중 시그널.
Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.mhm.config import MHMConfig


def generate_signals(df: pd.DataFrame, config: MHMConfig) -> StrategySignals:
    """MHM 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame.
        config: 전략 설정.

    Returns:
        StrategySignals.
    """
    from src.strategy.mhm.config import ShortMode

    # Shift(1): 전봉 기준
    max_agreement = df["max_agreement"].shift(1)
    pos_agreement = df["pos_agreement"].shift(1)
    neg_agreement = df["neg_agreement"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # Agreement filter: 최소 N개 horizon 동의
    enough_agreement = max_agreement >= config.agreement_threshold

    # Direction: weighted_mom 부호 + agreement 방향
    long_signal = enough_agreement & (pos_agreement >= config.agreement_threshold)
    short_signal = enough_agreement & (neg_agreement >= config.agreement_threshold)

    direction = _compute_direction(long_signal, short_signal, df, config)

    # Strength: direction * vol_scalar * conviction
    # conviction = agreement / 5 (최대 1.0)
    conviction = (max_agreement / 5.0).clip(upper=1.0).fillna(0.0)
    strength = direction.astype(float) * vol_scalar.fillna(0.0) * conviction

    if config.short_mode == ShortMode.HEDGE_ONLY:
        strength = pd.Series(
            np.where(direction == -1, strength * config.hedge_strength_ratio, strength),
            index=df.index,
        )

    strength = strength.fillna(0.0)

    # Entries / Exits
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
    config: MHMConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.mhm.config import ShortMode

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
