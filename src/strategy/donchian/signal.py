"""Donchian Channel Signal Generation.

터틀 트레이딩 규칙에 따른 Entry/Exit 시그널 생성.

Turtle Trading Rules:
    - Long Entry: close > entry_upper (N일 최고가 돌파)
    - Long Exit: close < exit_lower (M일 최저가 터치)
    - Short Entry: close < entry_lower (N일 최저가 돌파)
    - Short Exit: close > exit_upper (M일 최고가 터치)

Rules Applied:
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.donchian.config import ShortMode
from src.strategy.types import Direction, StrategySignals

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.strategy.donchian.config import DonchianConfig

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
        prev_pos = position[i - 1]
        position[i] = _get_next_position(
            prev_pos,
            long_entry[i],
            short_entry[i],
            long_exit[i],
            short_exit[i],
            allow_short,
        )

    return position


def _get_next_position(
    prev_pos: int,
    long_entry: bool,
    short_entry: bool,
    long_exit: bool,
    short_exit: bool,
    allow_short: bool,
) -> int:
    """다음 포지션 상태 결정.

    Args:
        prev_pos: 이전 포지션 (-1, 0, 1)
        long_entry: Long 진입 시그널
        short_entry: Short 진입 시그널
        long_exit: Long 청산 시그널
        short_exit: Short 청산 시그널
        allow_short: 숏 허용 여부

    Returns:
        다음 포지션 상태
    """
    # 상태 전이 테이블 기반 로직
    next_pos = prev_pos  # 기본값: 이전 상태 유지

    # Long Entry가 가장 우선순위 높음
    if long_entry and prev_pos != Direction.LONG.value:
        next_pos = Direction.LONG.value
    # Short Entry (숏 허용 시)
    elif short_entry and allow_short and prev_pos != Direction.SHORT.value:
        next_pos = Direction.SHORT.value
    # Exit 조건
    elif (prev_pos == Direction.LONG.value and long_exit) or (
        prev_pos == Direction.SHORT.value and short_exit
    ):
        next_pos = Direction.NEUTRAL.value

    return next_pos


def generate_signals(
    df: pd.DataFrame,
    config: DonchianConfig,
) -> StrategySignals:
    """Donchian Channel 시그널 생성.

    터틀 트레이딩 규칙:
        - Long Entry: 현재 종가 > 전봉 N일 최고가
        - Long Exit: 현재 종가 < 전봉 M일 최저가
        - Short Entry: 현재 종가 < 전봉 N일 최저가
        - Short Exit: 현재 종가 > 전봉 M일 최고가

    Args:
        df: 전처리된 DataFrame (preprocess 출력)
        config: 전략 설정

    Returns:
        StrategySignals NamedTuple
    """
    # 필수 컬럼 검증
    required_cols = {
        "close",
        "entry_upper",
        "entry_lower",
        "exit_upper",
        "exit_lower",
        "vol_scalar",
    }
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 컬럼 추출
    close: pd.Series = df["close"]  # type: ignore[assignment]
    entry_upper: pd.Series = df["entry_upper"]  # type: ignore[assignment]
    entry_lower: pd.Series = df["entry_lower"]  # type: ignore[assignment]
    exit_upper: pd.Series = df["exit_upper"]  # type: ignore[assignment]
    exit_lower: pd.Series = df["exit_lower"]  # type: ignore[assignment]
    vol_scalar: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    # Shift(1): 전봉 채널 기준 (미래 참조 방지)
    prev_entry_upper = entry_upper.shift(1)
    prev_entry_lower = entry_lower.shift(1)
    prev_exit_upper = exit_upper.shift(1)
    prev_exit_lower = exit_lower.shift(1)

    # Entry/Exit 시그널 (raw)
    long_entry_raw = close > prev_entry_upper
    short_entry_raw = close < prev_entry_lower
    long_exit_raw = close < prev_exit_lower
    short_exit_raw = close > prev_exit_upper

    # 상태 머신으로 포지션 계산
    position = _compute_position_state(
        long_entry_raw.to_numpy(),
        short_entry_raw.to_numpy(),
        long_exit_raw.to_numpy(),
        short_exit_raw.to_numpy(),
        allow_short=(config.short_mode == ShortMode.FULL),
    )
    direction = pd.Series(position, index=df.index, name="direction")

    # Entry/Exit 시그널 생성
    prev_direction = direction.shift(1).fillna(Direction.NEUTRAL.value).astype(int)

    long_entry = (direction == Direction.LONG.value) & (prev_direction != Direction.LONG.value)
    short_entry = (direction == Direction.SHORT.value) & (prev_direction != Direction.SHORT.value)
    entries = pd.Series(long_entry | short_entry, index=df.index, name="entries")

    to_neutral = (direction == Direction.NEUTRAL.value) & (
        prev_direction != Direction.NEUTRAL.value
    )
    reversal = (direction * prev_direction) < 0
    exits = pd.Series(to_neutral | reversal, index=df.index, name="exits")

    # 시그널 강도 계산
    strength = pd.Series(vol_scalar * direction.astype(float), index=df.index, name="strength")
    strength = strength.fillna(0.0)

    # 시그널 통계 로깅
    long_entries = int(long_entry.sum())
    short_entries = int(short_entry.sum())
    total_exits = int(exits.sum())

    if long_entries > 0 or short_entries > 0:
        logger.info(
            "📊 Donchian Signals | Long: %d, Short: %d, Exits: %d",
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
