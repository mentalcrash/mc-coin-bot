"""Flow Imbalance Signal Generator.

OFI + VPIN 기반 주문 흐름 방향 추종.

Signal Formula:
    1. Shift(1) 적용: ofi, vpin_proxy, vol_scalar
    2. Long: ofi_prev > entry_threshold & vpin_prev > vpin_threshold
    3. Short: ofi_prev < -entry_threshold & vpin_prev > vpin_threshold
    4. Exit: |ofi_prev| < exit_threshold OR timeout
    5. strength = direction * vol_scalar_prev

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.flow_imbalance.config import FlowImbalanceConfig, ShortMode
from src.strategy.types import Direction, StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: FlowImbalanceConfig | None = None,
) -> StrategySignals:
    """Flow Imbalance 시그널 생성.

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
        config: Flow Imbalance 설정. None이면 기본 설정 사용.

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    if config is None:
        config = FlowImbalanceConfig()

    required_cols = {"ofi", "vpin_proxy", "vol_scalar"}
    if config.short_mode == ShortMode.HEDGE_ONLY:
        required_cols.add("drawdown")

    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 1. Shift(1) 적용: 전봉 기준 시그널
    ofi_prev: pd.Series = df["ofi"].shift(1)  # type: ignore[assignment]
    vpin_prev: pd.Series = df["vpin_proxy"].shift(1)  # type: ignore[assignment]
    vol_scalar_prev: pd.Series = df["vol_scalar"].shift(1)  # type: ignore[assignment]

    # 2. Entry/Exit conditions
    vpin_active = vpin_prev > config.vpin_threshold
    long_entry_cond = (ofi_prev > config.ofi_entry_threshold) & vpin_active
    short_entry_cond = (ofi_prev < -config.ofi_entry_threshold) & vpin_active
    exit_cond = ofi_prev.abs() < config.ofi_exit_threshold

    # 3. Raw direction: entry → direction, exit → 0
    raw_direction = pd.Series(
        np.where(
            exit_cond,
            0,
            np.where(long_entry_cond, 1, np.where(short_entry_cond, -1, np.nan)),
        ),
        index=df.index,
    )

    # Forward-fill to hold position (NaN = maintain previous)
    raw_direction = raw_direction.ffill().fillna(0).astype(int)

    # 4. Timeout: count consecutive bars in same direction
    direction_change = raw_direction != raw_direction.shift(1)
    direction_group = direction_change.cumsum()
    bars_in_position = direction_group.groupby(direction_group).cumcount()

    # Apply timeout: force to neutral after timeout_bars
    timed_out = (bars_in_position >= config.timeout_bars) & (raw_direction != 0)
    raw_direction = raw_direction.where(~timed_out, 0)

    # 5. Strength = direction * vol_scalar
    strength_raw = raw_direction * vol_scalar_prev

    # 6. Direction 정규화
    direction = pd.Series(
        np.sign(strength_raw).fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    # 7. 강도 계산
    strength = pd.Series(
        strength_raw.fillna(0),
        index=df.index,
        name="strength",
    )

    # 8. 숏 모드에 따른 시그널 처리
    if config.short_mode == ShortMode.DISABLED:
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        drawdown_series: pd.Series = df["drawdown"].shift(1)  # type: ignore[assignment]
        hedge_active = drawdown_series < config.hedge_threshold

        short_mask = direction == Direction.SHORT
        suppress_short = short_mask & ~hedge_active
        direction = direction.where(~suppress_short, Direction.NEUTRAL)
        strength = strength.where(~suppress_short, 0.0)

        active_short = short_mask & hedge_active
        strength = strength.where(
            ~active_short,
            strength * config.hedge_strength_ratio,
        )

        hedge_bars = int(hedge_active.sum())
        if hedge_bars > 0:
            logger.info(
                "Hedge Mode | Active: {} bars ({:.1f}%), Threshold: {:.1f}%",
                hedge_bars,
                hedge_bars / len(hedge_active) * 100,
                config.hedge_threshold * 100,
            )

    # 9. 진입/청산 시그널
    prev_direction = direction.shift(1).fillna(0)

    long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)
    short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)

    entries = pd.Series(
        long_entry | short_entry,
        index=df.index,
        name="entries",
    )

    to_neutral = (direction == Direction.NEUTRAL) & (prev_direction != Direction.NEUTRAL)
    reversal = direction * prev_direction < 0

    exits = pd.Series(
        to_neutral | reversal,
        index=df.index,
        name="exits",
    )

    # 디버그: 시그널 통계
    valid_strength = strength[strength != 0]
    long_signals = strength[strength > 0]
    short_signals = strength[strength < 0]

    if len(valid_strength) > 0:
        logger.info(
            "Signal Statistics | Total: {} signals, Long: {} ({:.1f}%), Short: {} ({:.1f}%)",
            len(valid_strength),
            len(long_signals),
            len(long_signals) / len(valid_strength) * 100,
            len(short_signals),
            len(short_signals) / len(valid_strength) * 100,
        )

    return StrategySignals(
        entries=entries,
        exits=exits,
        direction=direction,
        strength=strength,
    )
