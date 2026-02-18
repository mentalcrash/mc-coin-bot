"""Liquidation Cascade Reversal 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
Multi-phase state machine: Buildup → Cascade → Reversal Entry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.liq_cascade_rev.config import LiqCascadeRevConfig


def generate_signals(df: pd.DataFrame, config: LiqCascadeRevConfig) -> StrategySignals:
    """Liquidation Cascade Reversal 시그널 생성.

    Phase 1 (Buildup): |fr_z| > buildup AND rv_ratio < 1.0
    Phase 2 (Cascade): |return| > N x ATR AND rv_ratio > expansion AND sign matches FR
    Phase 3 (Reversal): cascade 후 confirmation bars + body recovery
    Direction: 캐스케이드 반대 방향
    Exit: max_hold

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    # --- Shift(1): 전봉 기준 시그널 ---
    fr_z: pd.Series = df["fr_z"].shift(1)  # type: ignore[assignment]
    rv_ratio: pd.Series = df["rv_ratio"].shift(1)  # type: ignore[assignment]
    return_atr: pd.Series = df["return_atr_ratio"].shift(1)  # type: ignore[assignment]
    return_dir: pd.Series = df["return_dir"].shift(1)  # type: ignore[assignment]
    body_recovery: pd.Series = df["body_recovery"].shift(1)  # type: ignore[assignment]
    vol_scalar: pd.Series = df["vol_scalar"].shift(1)  # type: ignore[assignment]

    # --- State Machine ---
    direction = _cascade_reversal_direction(
        fr_z=fr_z,
        rv_ratio=rv_ratio,
        return_atr=return_atr,
        return_dir=return_dir,
        body_recovery=body_recovery,
        config=config,
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


def _cascade_reversal_direction(
    fr_z: pd.Series,
    rv_ratio: pd.Series,
    return_atr: pd.Series,
    return_dir: pd.Series,
    body_recovery: pd.Series,
    config: LiqCascadeRevConfig,
    index: pd.Index,
) -> pd.Series:
    """Multi-phase cascade reversal state machine.

    States: IDLE → BUILDUP → CASCADE_DETECTED → CONFIRMING → IN_POSITION
    """
    from src.strategy.liq_cascade_rev.config import ShortMode

    n = len(index)
    fr_arr = fr_z.to_numpy(dtype=float, na_value=0)
    rv_arr = rv_ratio.to_numpy(dtype=float, na_value=1)
    ret_atr_arr = return_atr.to_numpy(dtype=float, na_value=0)
    ret_dir_arr = return_dir.to_numpy(dtype=int, na_value=0)
    body_arr = body_recovery.to_numpy(dtype=float, na_value=0)

    direction = np.zeros(n, dtype=int)
    pos = 0
    hold_count = 0
    buildup_active = False
    cascade_dir = 0  # direction of the cascade move
    confirm_count = 0

    for i in range(n):
        # --- Existing position management ---
        if pos != 0:
            hold_count += 1
            if hold_count > config.max_hold_bars:
                pos = 0
                hold_count = 0
            direction[i] = pos
            continue

        # --- Phase 1: Buildup detection ---
        fr_extreme = abs(fr_arr[i]) > config.fr_buildup_threshold
        vol_calm = rv_arr[i] < 1.0
        if fr_extreme and vol_calm:
            buildup_active = True

        # --- Phase 2: Cascade detection ---
        if buildup_active and cascade_dir == 0:
            big_move = ret_atr_arr[i] > config.cascade_return_multiplier
            vol_spike = rv_arr[i] > config.vol_expansion_ratio
            # Cascade direction should align with FR stress
            fr_sign = 1 if fr_arr[i] > 0 else -1
            move_aligns = ret_dir_arr[i] == fr_sign
            if big_move and vol_spike and move_aligns:
                cascade_dir = ret_dir_arr[i]
                confirm_count = 0

        # --- Phase 3: Reversal confirmation ---
        if cascade_dir != 0:
            confirm_count += 1
            if confirm_count > config.reversal_confirmation_bars:
                if body_arr[i] > config.recovery_body_pct:
                    # Enter opposite to cascade
                    reversal_dir = -cascade_dir
                    if reversal_dir == -1 and config.short_mode == ShortMode.DISABLED:
                        reversal_dir = 0
                    pos = reversal_dir
                    hold_count = 1 if pos != 0 else 0
                    # Reset state
                    buildup_active = False
                    cascade_dir = 0
                    confirm_count = 0
                elif confirm_count > config.reversal_confirmation_bars + 3:
                    # Timeout: reset without entry
                    buildup_active = False
                    cascade_dir = 0
                    confirm_count = 0

        direction[i] = pos

    return pd.Series(direction, index=index, dtype=int)
