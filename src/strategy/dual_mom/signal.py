"""Dual Momentum Signal Generator.

Per-symbol momentum signal 생성. Cross-sectional ranking은
IntraPodAllocator(DUAL_MOMENTUM)에서 수행.

Signal: sign(rolling_return) * vol_scalar, shift(1), Long-Only 기본.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.dual_mom.config import DualMomConfig
from src.strategy.tsmom.config import ShortMode

if TYPE_CHECKING:
    from src.strategy.types import StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: DualMomConfig | None = None,
) -> StrategySignals:
    """Dual Momentum 시그널 생성.

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
        config: DualMom 설정

    Returns:
        StrategySignals NamedTuple
    """
    from src.strategy.types import Direction, StrategySignals

    if config is None:
        config = DualMomConfig()

    required_cols = {"rolling_return", "vol_scalar"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    rolling_return_series: pd.Series = df["rolling_return"]  # type: ignore[assignment]
    vol_scalar_series: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    # 1. Raw signal: sign(rolling_return) * vol_scalar
    return_direction = pd.Series(np.sign(rolling_return_series), index=df.index)
    raw_signal: pd.Series = return_direction * vol_scalar_series  # type: ignore[assignment]

    # 2. Shift(1): 미래 참조 편향 방지
    signal_shifted: pd.Series = raw_signal.shift(1)  # type: ignore[assignment]

    # 3. Direction
    direction_raw = pd.Series(np.sign(signal_shifted), index=df.index)
    direction = pd.Series(
        direction_raw.fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    # 4. Strength
    strength = pd.Series(
        signal_shifted.fillna(0),
        index=df.index,
        name="strength",
    )

    # 5. ShortMode 처리
    if config.short_mode == ShortMode.DISABLED:
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    # 6. Entry/Exit 시그널
    prev_direction = direction.shift(1).fillna(0)

    long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)
    short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)
    entries = pd.Series(long_entry | short_entry, index=df.index, name="entries")

    to_neutral = (direction == Direction.NEUTRAL) & (prev_direction != Direction.NEUTRAL)
    reversal = direction * prev_direction < 0
    exits = pd.Series(to_neutral | reversal, index=df.index, name="exits")

    valid_strength = strength[strength != 0]
    if len(valid_strength) > 0:
        long_count = int((strength > 0).sum())
        short_count = int((strength < 0).sum())
        logger.info(
            "DualMom Signal | Total: %d, Long: %d, Short: %d",
            len(valid_strength),
            long_count,
            short_count,
        )

    return StrategySignals(
        entries=entries,
        exits=exits,
        direction=direction,
        strength=strength,
    )
