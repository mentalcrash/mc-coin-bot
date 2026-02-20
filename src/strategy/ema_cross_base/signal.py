"""EMA Cross Base 시그널 생성."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.ema_cross_base.config import EmaCrossBaseConfig, ShortMode
from src.strategy.types import StrategySignals


def generate_signals(df: pd.DataFrame, config: EmaCrossBaseConfig) -> StrategySignals:
    """순수 EMA 크로스오버 시그널을 생성한다.

    Shift(1) Rule: 전봉 기준으로 시그널 생성 (lookahead bias 방지).

    Args:
        df: preprocess()로 지표가 추가된 DataFrame.
        config: 전략 설정.

    Returns:
        StrategySignals (entries, exits, direction, strength).
    """
    # Shift(1): 전봉 데이터 기준
    cross = df["ema_cross"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    long_signal = cross > 0
    short_signal = cross < 0

    direction = _compute_direction(long_signal, short_signal, config)

    # Strength = |direction| * vol_scalar
    strength = direction.astype(float) * vol_scalar.fillna(0)

    # Entry/Exit 감지
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
    config: EmaCrossBaseConfig,
) -> pd.Series:
    """ShortMode에 따른 방향 결정."""
    if config.short_mode == ShortMode.DISABLED:
        raw = np.where(long_signal, 1, 0)
    elif config.short_mode == ShortMode.HEDGE_ONLY:
        # HEDGE_ONLY: long + short 허용 (단, 동시 보유 아님)
        raw = np.where(long_signal, 1, np.where(short_signal, -1, 0))
    else:  # FULL
        raw = np.where(long_signal, 1, np.where(short_signal, -1, 0))

    return pd.Series(raw, index=long_signal.index, dtype=int)
