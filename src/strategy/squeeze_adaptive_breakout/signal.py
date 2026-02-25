"""Squeeze-Adaptive Breakout 시그널 생성.

Squeeze(BB inside KC) 해제 시점에 KAMA 적응적 방향 + BB position 확인으로
breakout 방향과 conviction을 결정한다.

Signal Logic:
    1. Squeeze fire: squeeze_lookback bars 이상 squeeze ON 유지 후 OFF 전환
    2. KAMA direction: close > KAMA → long bias, close < KAMA → short bias
    3. BB position: upper zone(>0.7) → long confirmation, lower zone(<0.3) → short confirmation
    4. Entry: squeeze fire & KAMA direction & BB position confirmation
    5. Exit: direction 변경 또는 중립 전환

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.squeeze_adaptive_breakout.config import ShortMode
from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.squeeze_adaptive_breakout.config import SqueezeAdaptiveBreakoutConfig


def generate_signals(
    df: pd.DataFrame,
    config: SqueezeAdaptiveBreakoutConfig,
) -> StrategySignals:
    """Squeeze-Adaptive Breakout 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    # --- Shift(1): 전봉 기준 시그널 ---
    squeeze_on_prev = df["squeeze_on"].shift(1).fillna(False).astype(bool)
    squeeze_on_prev2 = df["squeeze_on"].shift(2).fillna(False).astype(bool)
    squeeze_duration_prev = df["squeeze_duration"].shift(1).fillna(0)
    kama_prev = df["kama"].shift(1)
    close_prev = df["close"].shift(1)
    bb_pos_prev = df["bb_position"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- 1. Squeeze Fire: squeeze ON 지속 후 OFF 전환 ---
    # 2봉 전에 squeeze ON이었고, 1봉 전에 squeeze OFF로 전환
    # 추가로, squeeze가 설정된 lookback 이상 지속되었어야 함
    squeeze_fire = (
        squeeze_on_prev2
        & ~squeeze_on_prev
        & (squeeze_duration_prev.shift(1) >= config.squeeze_lookback)
    )

    # --- 2. KAMA Adaptive Direction ---
    kama_long = close_prev > kama_prev
    kama_short = close_prev < kama_prev

    # --- 3. BB Position Confirmation ---
    bb_pos_long = bb_pos_prev > config.bb_pos_long_threshold
    bb_pos_short = bb_pos_prev < config.bb_pos_short_threshold

    # --- 4. Combined Signal ---
    long_signal = squeeze_fire & kama_long & bb_pos_long
    short_signal = squeeze_fire & kama_short & bb_pos_short

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength: direction * vol_scalar * conviction ---
    # BB position을 conviction으로 사용: long은 bb_pos, short은 (1 - bb_pos)
    conviction = pd.Series(
        np.where(
            direction == 1,
            bb_pos_prev.fillna(0.5),
            np.where(direction == -1, 1.0 - bb_pos_prev.fillna(0.5), 0.0),
        ),
        index=df.index,
    )

    strength = direction.astype(float) * vol_scalar.fillna(0) * conviction.clip(lower=0.1)

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
    config: SqueezeAdaptiveBreakoutConfig,
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
