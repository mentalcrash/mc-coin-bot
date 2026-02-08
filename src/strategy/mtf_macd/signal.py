"""MTF MACD Signal Generator.

MACD crossover 기반 진입/청산 시그널을 생성합니다.

Signal Logic:
    - Long Entry: MACD line crosses above signal line AND MACD > 0 (bullish trend)
    - Short Entry: MACD line crosses below signal line AND MACD < 0 (bearish trend)
    - Long Exit: bearish candle (close < open)
    - Short Exit: bullish candle (close > open)

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops, except state machine)
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.mtf_macd.config import ShortMode
from src.strategy.types import Direction, StrategySignals

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.strategy.mtf_macd.config import MtfMacdConfig

logger = logging.getLogger(__name__)


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
    config: MtfMacdConfig,
) -> StrategySignals:
    """MTF MACD 시그널 생성.

    전처리된 DataFrame에서 MACD crossover 기반 진입/청산 시그널을 생성합니다.
    Shift(1) Rule을 적용하여 미래 참조 편향을 방지합니다.

    Signal Generation Pipeline:
        1. MACD crossover 감지 (shift(1), shift(2) 사용)
        2. Trend filter 적용 (MACD > 0 = bullish, MACD < 0 = bearish)
        3. Exit 조건: bearish/bullish candle (shift(1) 적용)
        4. 상태 머신으로 포지션 추적
        5. ShortMode 처리 (DISABLED/FULL)
        6. Vol scalar로 strength 계산

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: macd_line, signal_line, open, close, vol_scalar
        config: MTF MACD 설정

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    # 입력 검증
    required_cols = {"macd_line", "signal_line", "open", "close", "vol_scalar"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # ================================================================
    # 1. MACD Crossover 감지 (Shift(1) Rule 적용)
    # ================================================================
    macd_line: pd.Series = df["macd_line"]  # type: ignore[assignment]
    signal_line: pd.Series = df["signal_line"]  # type: ignore[assignment]
    open_price: pd.Series = df["open"]  # type: ignore[assignment]
    close_price: pd.Series = df["close"]  # type: ignore[assignment]
    vol_scalar: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    # Shift(1): 전봉, Shift(2): 전전봉 기준 (미래 참조 방지)
    prev_macd_1: pd.Series = macd_line.shift(1)  # type: ignore[assignment]
    prev_signal_1: pd.Series = signal_line.shift(1)  # type: ignore[assignment]
    prev_macd_2: pd.Series = macd_line.shift(2)  # type: ignore[assignment]
    prev_signal_2: pd.Series = signal_line.shift(2)  # type: ignore[assignment]

    # Long Entry: MACD crosses above Signal (전전봉 MACD <= Signal, 전봉 MACD > Signal)
    #   AND trend bullish (전봉 MACD > 0)
    cross_up = (prev_macd_2 <= prev_signal_2) & (prev_macd_1 > prev_signal_1)
    trend_bull = prev_macd_1 > 0
    long_entry_raw = cross_up & trend_bull

    # Short Entry: MACD crosses below Signal (전전봉 MACD >= Signal, 전봉 MACD < Signal)
    #   AND trend bearish (전봉 MACD < 0)
    cross_down = (prev_macd_2 >= prev_signal_2) & (prev_macd_1 < prev_signal_1)
    trend_bear = prev_macd_1 < 0
    short_entry_raw = cross_down & trend_bear

    # ================================================================
    # 2. Exit 조건 (candle color, shift(1) 적용)
    # ================================================================
    # Long Exit: 전봉이 bearish candle (close < open)
    prev_close: pd.Series = close_price.shift(1)  # type: ignore[assignment]
    prev_open: pd.Series = open_price.shift(1)  # type: ignore[assignment]

    long_exit_raw = prev_close < prev_open  # bearish candle
    short_exit_raw = prev_close > prev_open  # bullish candle

    # NaN을 False로 채우기 (shift로 인한 NaN)
    long_entry_raw = long_entry_raw.fillna(False)
    short_entry_raw = short_entry_raw.fillna(False)
    long_exit_raw = long_exit_raw.fillna(False)
    short_exit_raw = short_exit_raw.fillna(False)

    # ================================================================
    # 3. 상태 머신으로 포지션 추적
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
    # 4. Strength 계산 (direction * vol_scalar)
    # ================================================================
    strength = pd.Series(
        direction.astype(float) * vol_scalar.fillna(0),
        index=df.index,
        name="strength",
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
        logger.info(
            "MTF MACD Signals | Long: %d, Short: %d, Exits: %d",
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
