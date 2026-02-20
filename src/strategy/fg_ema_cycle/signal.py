"""F&G EMA Long-Cycle 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.

장기 F&G EMA의 크로스오버로 매크로 사이클 방향 포착.
Fast EMA crosses above Slow EMA (fear zone) → long.
Fast EMA crosses below Slow EMA (greed zone) → short.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.fg_ema_cycle.config import FgEmaCycleConfig


def generate_signals(df: pd.DataFrame, config: FgEmaCycleConfig) -> StrategySignals:
    """F&G EMA Long-Cycle 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.fg_ema_cycle.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    fg_ema_fast = df["fg_ema_fast"].shift(1)
    fg_ema_slow = df["fg_ema_slow"].shift(1)
    cycle_position = df["cycle_position"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # 전전봉 (크로스오버 감지용)
    fg_ema_fast_prev = df["fg_ema_fast"].shift(2)
    fg_ema_slow_prev = df["fg_ema_slow"].shift(2)

    # --- Crossover Detection ---
    # Fast crosses above Slow (상승 크로스)
    golden_cross = (fg_ema_fast > fg_ema_slow) & (fg_ema_fast_prev <= fg_ema_slow_prev)

    # Fast crosses below Slow (하락 크로스)
    death_cross = (fg_ema_fast < fg_ema_slow) & (fg_ema_fast_prev >= fg_ema_slow_prev)

    # --- Zone Filter: 크로스 발생 시 slow EMA의 위치 확인 ---
    long_signal = golden_cross & (fg_ema_slow < config.fear_cycle)
    short_signal = death_cross & (fg_ema_slow > config.greed_cycle)

    # --- Direction (forward-fill: 반대 시그널까지 hold) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength: cycle_position 극단일수록 강함 ---
    cycle_strength = cycle_position.abs().clip(upper=1.0).fillna(0)
    strength = direction.astype(float) * vol_scalar.fillna(0) * np.maximum(cycle_strength, 0.5)

    if config.short_mode == ShortMode.HEDGE_ONLY:
        strength = pd.Series(
            np.where(direction == -1, strength * config.hedge_strength_ratio, strength),
            index=df.index,
        )

    strength = pd.Series(strength, index=df.index).fillna(0.0)

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
    config: FgEmaCycleConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산.

    크로스오버 이벤트 후 반대 시그널까지 hold (forward-fill).
    """
    from src.strategy.fg_ema_cycle.config import ShortMode

    n = len(df)
    event = np.where(long_signal, 1, np.where(short_signal, -1, 0))

    # Forward-fill: 마지막 이벤트 방향을 유지
    raw = np.zeros(n, dtype=int)
    last_dir = 0
    for i in range(n):
        if event[i] != 0:
            last_dir = int(event[i])
        raw[i] = last_dir

    # ShortMode 적용
    if config.short_mode == ShortMode.DISABLED:
        raw = np.where(raw == -1, 0, raw)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        dd = df["drawdown"].shift(1).to_numpy()
        hedge_active = dd < config.hedge_threshold
        raw = np.where(
            (raw == -1) & ~hedge_active,
            0,
            raw,
        )

    return pd.Series(raw, index=df.index, dtype=int)
