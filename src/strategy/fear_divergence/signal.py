"""Fear-Greed Divergence 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
F&G 극단 이벤트 기반 → carry-forward position until F&G 정상 복귀.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

_FG_EXIT_LOW = 30
_FG_EXIT_HIGH = 70

if TYPE_CHECKING:
    from src.strategy.fear_divergence.config import FearDivergenceConfig


def generate_signals(df: pd.DataFrame, config: FearDivergenceConfig) -> StrategySignals:
    """Fear-Greed Divergence 시그널 생성.

    fear_extreme: fg < threshold AND fg < fg_ma - deviation
    greed_extreme: fg > threshold AND fg > fg_ma + deviation
    fear_divergence: fear_extreme AND price_roc > 0 (반전 시작)
    greed_divergence: greed_extreme AND price_roc < 0
    Entry LONG: fear_divergence AND er > er_min
    Entry SHORT: greed_divergence AND er > er_min (HEDGE_ONLY)
    Exit: fg 30-70 복귀

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    # --- Shift(1): 전봉 기준 시그널 ---
    fg: pd.Series = df["oc_fear_greed"].shift(1)  # type: ignore[assignment]
    fg_ma = df["fg_ma"].shift(1)
    price_roc = df["price_roc"].shift(1)
    er = df["er"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Extreme Conditions ---
    fear_extreme = (fg < config.fg_fear_threshold) & (fg < fg_ma - config.fg_deviation)
    greed_extreme = (fg > config.fg_greed_threshold) & (fg > fg_ma + config.fg_deviation)

    # --- Divergence ---
    fear_div = fear_extreme & (price_roc > 0)
    greed_div = greed_extreme & (price_roc < 0)

    # --- Trend Confirmation ---
    trend_ok = er > config.er_min

    # --- Entry Signals ---
    long_signal = fear_div & trend_ok
    short_signal = greed_div & trend_ok

    # --- Exit: F&G 정상 복귀 ---
    exit_signal = (fg > _FG_EXIT_LOW) & (fg < _FG_EXIT_HIGH)

    # --- State Machine: carry-forward ---
    direction = _carry_forward_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        exit_signal=exit_signal,
        short_mode=config.short_mode,
        index=df.index,
    )

    # --- Strength ---
    base_scalar = vol_scalar.fillna(0)
    strength = direction.astype(float) * base_scalar
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


def _carry_forward_direction(
    long_signal: pd.Series,
    short_signal: pd.Series,
    exit_signal: pd.Series,
    short_mode: int,
    index: pd.Index,
) -> pd.Series:
    """F&G divergence state machine: carry-forward until F&G normalizes."""
    from src.strategy.fear_divergence.config import ShortMode

    n = len(index)
    long_arr = long_signal.to_numpy(dtype=bool, na_value=False)
    short_arr = short_signal.to_numpy(dtype=bool, na_value=False)
    exit_arr = exit_signal.to_numpy(dtype=bool, na_value=False)

    direction = np.zeros(n, dtype=int)
    pos = 0

    for i in range(n):
        if exit_arr[i] and pos != 0:
            pos = 0
        elif long_arr[i] and pos <= 0:
            pos = 1
        elif short_arr[i] and pos >= 0:
            pos = 0 if short_mode == ShortMode.DISABLED else -1
        direction[i] = pos

    return pd.Series(direction, index=index, dtype=int)
