"""Funding Rate + Stablecoin Confluence 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
FR 극단 이벤트 기반 → carry-forward position until exit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.fr_stab_conf.config import FrStabConfConfig


def generate_signals(df: pd.DataFrame, config: FrStabConfConfig) -> StrategySignals:
    """FR + Stablecoin Confluence 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """

    # --- Shift(1): 전봉 기준 시그널 ---
    fr_z = df["fr_z"].shift(1)
    stab_z = df["stab_z"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Raw Signals ---
    # FR 양수 극단 + Stablecoin 유출 → SHORT (과열 반전)
    short_signal = (fr_z > config.fr_short_threshold) & (stab_z < 0)
    # FR 음수 극단 + Stablecoin 유입 → LONG (과냉 반전)
    long_signal = (fr_z < config.fr_long_threshold) & (stab_z > 0)
    # 청산: FR 정상 복귀
    exit_signal = fr_z.abs() < config.fr_exit_threshold

    # --- State Machine: carry-forward position ---
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
    """FR 극단 이벤트 기반 state machine.

    진입 후 exit_signal까지 포지션 유지 (carry-forward).
    벡터화가 어려운 state machine이므로 numba-free numpy 구현.
    """
    from src.strategy.fr_stab_conf.config import ShortMode

    n = len(index)
    long_arr = long_signal.to_numpy(dtype=bool, na_value=False)
    short_arr = short_signal.to_numpy(dtype=bool, na_value=False)
    exit_arr = exit_signal.to_numpy(dtype=bool, na_value=False)

    direction = np.zeros(n, dtype=int)
    pos = 0  # current position

    for i in range(n):
        if exit_arr[i] and pos != 0:
            pos = 0
        elif long_arr[i]:
            pos = 1
        elif short_arr[i]:
            pos = 0 if short_mode == ShortMode.DISABLED else -1
        direction[i] = pos

    return pd.Series(direction, index=index, dtype=int)
