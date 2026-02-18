"""Vol Squeeze + Derivatives 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
Vol 압축 → expansion breakout + FR contrarian 확인.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.vol_squeeze_deriv.config import VolSqueezeDerivConfig


def generate_signals(df: pd.DataFrame, config: VolSqueezeDerivConfig) -> StrategySignals:
    """Vol Squeeze + Derivatives 시그널 생성.

    Phase 1: squeeze = vol_rank < threshold (duration tracking)
    Phase 2: expanding = atr_ratio > expansion_ratio
    Phase 3: direction = sma_direction, FR contrarian confirm
    Entry: squeeze_duration >= min_squeeze_bars AND expanding
    Exit: vol_rank > vol_exit_rank

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    # --- Shift(1): 전봉 기준 시그널 ---
    vol_rank: pd.Series = df["vol_rank"].shift(1)  # type: ignore[assignment]
    atr_ratio: pd.Series = df["atr_ratio"].shift(1)  # type: ignore[assignment]
    fr_z: pd.Series = df["fr_z"].shift(1)  # type: ignore[assignment]
    sma_dir: pd.Series = df["sma_direction"].shift(1)  # type: ignore[assignment]
    vol_scalar: pd.Series = df["vol_scalar"].shift(1)  # type: ignore[assignment]

    # --- Phase Detection ---
    squeeze = vol_rank < config.squeeze_threshold
    expanding = atr_ratio > config.expansion_ratio

    # --- State Machine: squeeze duration + breakout ---
    direction = _squeeze_breakout_direction(
        squeeze=squeeze,
        expanding=expanding,
        sma_dir=sma_dir,
        fr_z=fr_z,
        vol_rank=vol_rank,
        config=config,
        index=df.index,
    )

    # --- Strength: contrarian vs aligned weight ---
    is_contrarian = (direction == 1) & (fr_z > 0) | (direction == -1) & (fr_z < 0)
    weight_factor = pd.Series(config.aligned_weight, index=df.index)
    weight_factor = weight_factor.where(~is_contrarian, config.contrarian_weight)
    base_scalar = vol_scalar.fillna(0) * weight_factor
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


def _squeeze_breakout_direction(
    squeeze: pd.Series,
    expanding: pd.Series,
    sma_dir: pd.Series,
    fr_z: pd.Series,
    vol_rank: pd.Series,
    config: VolSqueezeDerivConfig,
    index: pd.Index,
) -> pd.Series:
    """Squeeze → expansion breakout state machine."""
    from src.strategy.vol_squeeze_deriv.config import ShortMode

    n = len(index)
    sq_arr = squeeze.to_numpy(dtype=bool, na_value=False)
    exp_arr = expanding.to_numpy(dtype=bool, na_value=False)
    sma_arr = sma_dir.to_numpy(dtype=float, na_value=0)
    vr_arr = vol_rank.to_numpy(dtype=float, na_value=50)

    direction = np.zeros(n, dtype=int)
    pos = 0
    squeeze_count = 0

    for i in range(n):
        # Track squeeze duration
        if sq_arr[i]:
            squeeze_count += 1
        else:
            squeeze_count = 0

        # Exit: vol overextended
        if pos != 0 and vr_arr[i] > config.vol_exit_rank:
            pos = 0

        # Entry: squeeze met + expansion + no existing position
        if pos == 0 and squeeze_count >= config.min_squeeze_bars and exp_arr[i]:
            raw_dir = int(np.sign(sma_arr[i])) if sma_arr[i] != 0 else 0
            if raw_dir == -1 and config.short_mode == ShortMode.DISABLED:
                raw_dir = 0
            pos = raw_dir
            squeeze_count = 0  # Reset after entry

        direction[i] = pos

    return pd.Series(direction, index=index, dtype=int)
