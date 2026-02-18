"""Vol Squeeze Breakout 시그널 생성.

스퀴즈 상태에서 BB 돌파 + 거래량 서지 → breakout 시그널.
Shift(1) Rule: squeeze 상태, BB 경계, 거래량 기준은 shift(1) 적용.
close는 현재 bar (breakout은 현재 bar에서 발생하는 이벤트).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals
from src.strategy.vol_squeeze_brk.config import ShortMode

if TYPE_CHECKING:
    from src.strategy.vol_squeeze_brk.config import VolSqueezeBrkConfig


def generate_signals(df: pd.DataFrame, config: VolSqueezeBrkConfig) -> StrategySignals:
    """Vol Squeeze Breakout 시그널 생성.

    Signal Logic:
        1. squeeze_prev = in_squeeze(shifted) — 전봉 스퀴즈 상태
        2. breakout_up = close > bb_upper(shifted) & squeeze_prev
        3. breakout_down = close < bb_lower(shifted) & squeeze_prev
        4. vol_surge = volume > vol_avg(shifted) * multiplier
        5. direction = breakout & vol_surge

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    close: pd.Series = df["close"]  # type: ignore[assignment]
    volume: pd.Series = df["volume"]  # type: ignore[assignment]

    # --- Shift(1): 전봉 기준 squeeze/BB/vol ---
    squeeze_prev = df["in_squeeze"].astype(bool).shift(1, fill_value=False)
    bb_upper_prev = df["bb_upper"].shift(1)
    bb_lower_prev = df["bb_lower"].shift(1)
    vol_avg_prev = df["vol_avg"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    # 1. Breakout detection (current close vs prev BB boundaries)
    breakout_up = (close > bb_upper_prev) & squeeze_prev
    breakout_down = (close < bb_lower_prev) & squeeze_prev

    # 2. Volume surge confirmation
    vol_surge = volume > (vol_avg_prev * config.vol_surge_multiplier)

    # 3. Confirmed breakouts
    long_signal = breakout_up & vol_surge
    short_signal = breakout_down & vol_surge

    # --- Direction (ShortMode 3-way) ---
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
    config: VolSqueezeBrkConfig,
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
