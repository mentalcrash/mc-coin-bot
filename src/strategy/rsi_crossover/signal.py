"""RSI Crossover Signal Generator.

RSI 크로스오버 기반 진입/청산 시그널을 생성합니다.

Signal Logic:
    - Long Entry: RSI가 entry_oversold(30)를 상향 크로스
    - Short Entry: RSI가 entry_overbought(70)를 하향 크로스
    - Long Exit: RSI가 exit_long(60) 도달
    - Short Exit: RSI가 exit_short(40) 도달

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops, except state machine)
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger as loguru_logger

from src.strategy.tsmom.config import ShortMode
from src.strategy.types import Direction, StrategySignals

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.strategy.rsi_crossover.config import RSICrossoverConfig


def _compute_position_state(
    long_entry: NDArray[np.bool_],
    short_entry: NDArray[np.bool_],
    long_exit: NDArray[np.bool_],
    short_exit: NDArray[np.bool_],
    allow_short: bool,
) -> NDArray[np.int32]:
    """상태 머신 기반 포지션 계산.

    Args:
        long_entry: Long 진입 시그널 배열
        short_entry: Short 진입 시그널 배열
        long_exit: Long 청산 시그널 배열
        short_exit: Short 청산 시그널 배열
        allow_short: 숏 허용 여부

    Returns:
        포지션 상태 배열 (-1, 0, 1)
    """
    n = len(long_entry)
    position = np.zeros(n, dtype=np.int32)

    for i in range(1, n):
        prev = position[i - 1]
        # Long entry (highest priority)
        if long_entry[i] and prev != Direction.LONG.value:
            position[i] = Direction.LONG.value
        # Short entry
        elif short_entry[i] and allow_short and prev != Direction.SHORT.value:
            position[i] = Direction.SHORT.value
        # Exit conditions
        elif (prev == Direction.LONG.value and long_exit[i]) or (
            prev == Direction.SHORT.value and short_exit[i]
        ):
            position[i] = Direction.NEUTRAL.value
        else:
            position[i] = prev

    return position


def generate_signals(
    df: pd.DataFrame,
    config: RSICrossoverConfig,
) -> StrategySignals:
    """RSI Crossover 시그널 생성.

    전처리된 DataFrame에서 RSI 크로스오버 기반 진입/청산 시그널을 생성합니다.
    Shift(1) Rule을 적용하여 미래 참조 편향을 방지합니다.

    Signal Generation Pipeline:
        1. RSI 크로스오버 감지 (shift(1), shift(2) 사용)
        2. 상태 머신으로 포지션 추적
        3. ShortMode 처리 (DISABLED/HEDGE_ONLY/FULL)
        4. Vol scalar로 strength 계산
        5. Entry/Exit 시그널 생성

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: rsi, vol_scalar
        config: RSI Crossover 설정

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    # 입력 검증
    required_cols = {"rsi", "vol_scalar"}
    if config.short_mode == ShortMode.HEDGE_ONLY:
        required_cols.add("drawdown")
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # ================================================================
    # 1. RSI 크로스오버 감지 (Shift(1) Rule 적용)
    # ================================================================
    rsi: pd.Series = df["rsi"]  # type: ignore[assignment]
    vol_scalar: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    # Shift(1): 전봉, Shift(2): 전전봉 기준 (미래 참조 방지)
    rsi_prev: pd.Series = rsi.shift(1)  # type: ignore[assignment]
    rsi_prev2: pd.Series = rsi.shift(2)  # type: ignore[assignment]

    # Long Entry: RSI가 entry_oversold(30)를 상향 크로스
    # 전전봉 <= 30 이고 전봉 > 30 → 크로스오버 발생
    long_entry_raw = (rsi_prev > config.entry_oversold) & (rsi_prev2 <= config.entry_oversold)

    # Short Entry: RSI가 entry_overbought(70)를 하향 크로스
    # 전전봉 >= 70 이고 전봉 < 70 → 크로스언더 발생
    short_entry_raw = (rsi_prev < config.entry_overbought) & (rsi_prev2 >= config.entry_overbought)

    # Long Exit: RSI가 exit_long(60) 도달
    long_exit_raw = rsi_prev > config.exit_long

    # Short Exit: RSI가 exit_short(40) 도달
    short_exit_raw = rsi_prev < config.exit_short

    # NaN을 False로 채우기 (shift로 인한 NaN)
    long_entry_raw = long_entry_raw.fillna(False)
    short_entry_raw = short_entry_raw.fillna(False)
    long_exit_raw = long_exit_raw.fillna(False)
    short_exit_raw = short_exit_raw.fillna(False)

    # ================================================================
    # 2. 상태 머신으로 포지션 추적
    # ================================================================
    allow_short = config.short_mode == ShortMode.FULL
    position = _compute_position_state(
        long_entry_raw.to_numpy(),
        short_entry_raw.to_numpy(),
        long_exit_raw.to_numpy(),
        short_exit_raw.to_numpy(),
        allow_short=allow_short,
    )
    direction = pd.Series(position, index=df.index, name="direction")

    # ================================================================
    # 3. ShortMode 처리 (HEDGE_ONLY)
    # ================================================================
    if config.short_mode == ShortMode.HEDGE_ONLY:
        drawdown_series: pd.Series = df["drawdown"]  # type: ignore[assignment]
        hedge_active = drawdown_series < config.hedge_threshold

        short_mask = direction == Direction.SHORT
        suppress_short = short_mask & ~hedge_active
        direction = direction.where(~suppress_short, Direction.NEUTRAL)

        active_short = short_mask & hedge_active
        hedge_days = int(hedge_active.sum())
        if hedge_days > 0:
            loguru_logger.info(
                "Hedge Mode | Active: {} days ({:.1f}%), Threshold: {:.1f}%",
                hedge_days,
                hedge_days / len(hedge_active) * 100,
                config.hedge_threshold * 100,
            )
            # active_short은 아래 strength 계산에서 사용
            _ = active_short

    # ================================================================
    # 4. Strength 계산 (direction * vol_scalar, shift 적용)
    # ================================================================
    # vol_scalar를 shift(1)하여 미래 참조 방지
    vol_scalar_shifted: pd.Series = vol_scalar.shift(1)  # type: ignore[assignment]

    # strength = direction * vol_scalar (방향 + 크기)
    strength = pd.Series(
        direction.astype(float) * vol_scalar_shifted.fillna(0),
        index=df.index,
        name="strength",
    )

    # HEDGE_ONLY 모드에서 헤지 활성 시 숏 강도 조절
    if config.short_mode == ShortMode.HEDGE_ONLY:
        drawdown_series = df["drawdown"]  # type: ignore[assignment]
        hedge_active = drawdown_series < config.hedge_threshold
        short_mask = direction == Direction.SHORT
        active_short = short_mask & hedge_active
        strength = strength.where(
            ~active_short,
            strength * config.hedge_strength_ratio,
        )

    strength = strength.fillna(0.0)

    # ================================================================
    # 5. Entry/Exit 시그널 생성
    # ================================================================
    prev_direction = direction.shift(1).fillna(Direction.NEUTRAL.value).astype(int)

    long_entry = (direction == Direction.LONG.value) & (prev_direction != Direction.LONG.value)
    short_entry = (direction == Direction.SHORT.value) & (prev_direction != Direction.SHORT.value)
    entries = pd.Series(long_entry | short_entry, index=df.index, name="entries")

    to_neutral = (direction == Direction.NEUTRAL.value) & (
        prev_direction != Direction.NEUTRAL.value
    )
    reversal = (direction * prev_direction) < 0
    exits = pd.Series(to_neutral | reversal, index=df.index, name="exits")

    # 시그널 통계 로깅
    long_entries = int(long_entry.sum())
    short_entries = int(short_entry.sum())
    total_exits = int(exits.sum())

    if long_entries > 0 or short_entries > 0:
        loguru_logger.info(
            "RSI Crossover Signals | Long: {}, Short: {}, Exits: {}",
            long_entries,
            short_entries,
            total_exits,
        )

    return StrategySignals(
        entries=entries,
        exits=exits,
        direction=direction,
        strength=strength,
    )
