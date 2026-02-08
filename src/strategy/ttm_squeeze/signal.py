"""TTM Squeeze Signal Generator.

Squeeze 해제(ON -> OFF) 시 momentum 방향으로 진입하고,
close가 exit SMA를 역방향 크로스 시 청산하는 시그널을 생성합니다.

Signal Logic:
    1. Squeeze fire: 2봉 전 squeeze ON, 1봉 전 squeeze OFF (해제)
    2. Entry: squeeze fire + momentum 방향 (long/short)
    3. Exit: close가 exit SMA 반대편 크로스
    4. State machine으로 포지션 추적

Rules Applied:
    - #12 Data Engineering: Vectorization (state machine은 예외적 loop)
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.ttm_squeeze.config import ShortMode
from src.strategy.types import Direction, StrategySignals

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.strategy.ttm_squeeze.config import TtmSqueezeConfig

logger = logging.getLogger(__name__)


def _compute_position_state(
    squeeze_fire: NDArray[np.bool_],
    mom_long: NDArray[np.bool_],
    mom_short: NDArray[np.bool_],
    long_exit: NDArray[np.bool_],
    short_exit: NDArray[np.bool_],
    allow_short: bool,
) -> NDArray[np.int32]:
    """상태 머신 기반 포지션 계산.

    Args:
        squeeze_fire: Squeeze 해제 시그널 배열
        mom_long: 양(+) momentum 배열
        mom_short: 음(-) momentum 배열
        long_exit: Long 청산 시그널 배열
        short_exit: Short 청산 시그널 배열
        allow_short: 숏 허용 여부

    Returns:
        포지션 상태 배열 (-1, 0, 1)
    """
    n = len(squeeze_fire)
    position = np.zeros(n, dtype=np.int32)

    for i in range(1, n):
        prev = position[i - 1]

        # Long entry: squeeze fires + positive momentum
        if squeeze_fire[i] and mom_long[i] and prev != Direction.LONG.value:
            position[i] = Direction.LONG.value
        # Short entry: squeeze fires + negative momentum
        elif squeeze_fire[i] and mom_short[i] and allow_short and prev != Direction.SHORT.value:
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
    config: TtmSqueezeConfig,
) -> StrategySignals:
    """TTM Squeeze 시그널 생성.

    전처리된 DataFrame에서 squeeze 해제 기반 진입/청산 시그널을 생성합니다.
    Shift(1) Rule을 적용하여 미래 참조 편향을 방지합니다.

    Signal Generation Pipeline:
        1. Squeeze fire 감지 (shift(1), shift(2) 사용)
        2. Momentum 방향 결정 (shift(1))
        3. Exit SMA 크로스 감지 (shift(1))
        4. 상태 머신으로 포지션 추적
        5. ShortMode 처리
        6. Vol scalar로 strength 계산
        7. Entry/Exit bool 시그널 생성

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: squeeze_on, momentum, exit_sma, close, vol_scalar
        config: TTM Squeeze 설정

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    # 입력 검증
    required_cols = {"squeeze_on", "momentum", "exit_sma", "close", "vol_scalar"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # ================================================================
    # 1. Squeeze Fire 감지 (Shift(1) Rule 적용)
    # ================================================================
    squeeze_on: pd.Series = df["squeeze_on"]  # type: ignore[assignment]
    momentum: pd.Series = df["momentum"]  # type: ignore[assignment]
    close: pd.Series = df["close"]  # type: ignore[assignment]
    exit_sma: pd.Series = df["exit_sma"]  # type: ignore[assignment]
    vol_scalar: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    # Shift(1): 전봉, Shift(2): 전전봉 기준 (미래 참조 방지)
    # shift()는 bool → object 변환하므로 직접 numpy로 처리
    sq_arr = squeeze_on.to_numpy().astype(bool)
    prev_sq = pd.Series(
        np.concatenate([[False], sq_arr[:-1]]),
        index=squeeze_on.index,
        dtype=bool,
    )
    prev_sq2 = pd.Series(
        np.concatenate([[False, False], sq_arr[:-2]]),
        index=squeeze_on.index,
        dtype=bool,
    )

    # Squeeze fire: 2봉 전에 squeeze ON, 1봉 전에 squeeze OFF (해제 전환)
    squeeze_fire_raw = prev_sq2 & ~prev_sq

    # ================================================================
    # 2. Momentum 방향 결정 (Shift(1) — 전봉 momentum 사용)
    # ================================================================
    prev_mom: pd.Series = momentum.shift(1)  # type: ignore[assignment]
    mom_long = prev_mom.fillna(0) > 0
    mom_short = prev_mom.fillna(0) < 0

    # ================================================================
    # 3. Exit SMA 크로스 감지 (Shift(1) — 전봉 기준)
    # ================================================================
    prev_close: pd.Series = close.shift(1)  # type: ignore[assignment]
    prev_sma: pd.Series = exit_sma.shift(1)  # type: ignore[assignment]

    # Long exit: 전봉에서 close가 SMA 아래
    long_exit_raw = prev_close.fillna(0) < prev_sma.fillna(0)
    # Short exit: 전봉에서 close가 SMA 위
    short_exit_raw = prev_close.fillna(0) > prev_sma.fillna(0)

    # NaN -> False
    squeeze_fire_raw = squeeze_fire_raw.fillna(False)
    long_exit_raw = long_exit_raw.fillna(False)
    short_exit_raw = short_exit_raw.fillna(False)

    # ================================================================
    # 4. 상태 머신으로 포지션 추적
    # ================================================================
    allow_short = config.short_mode == ShortMode.FULL
    position = _compute_position_state(
        squeeze_fire_raw.to_numpy().astype(bool),
        mom_long.to_numpy().astype(bool),
        mom_short.to_numpy().astype(bool),
        long_exit_raw.to_numpy().astype(bool),
        short_exit_raw.to_numpy().astype(bool),
        allow_short=allow_short,
    )
    direction = pd.Series(position, index=df.index, name="direction")

    # ================================================================
    # 5. ShortMode 처리 (DISABLED 모드에서 short 제거)
    # ================================================================
    if config.short_mode == ShortMode.DISABLED:
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)

    # ================================================================
    # 6. Strength 계산 (direction * vol_scalar)
    # ================================================================
    # vol_scalar는 이미 shift(1) 적용됨 (preprocessor에서)
    strength = pd.Series(
        direction.astype(float) * vol_scalar.fillna(0),
        index=df.index,
        name="strength",
    )

    # DISABLED 모드에서 음의 strength 제거
    if config.short_mode == ShortMode.DISABLED:
        strength = strength.clip(lower=0.0)

    strength = strength.fillna(0.0)

    # ================================================================
    # 7. Entry/Exit 시그널 생성
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
    squeeze_fires = int(squeeze_fire_raw.sum())

    if long_entries > 0 or short_entries > 0:
        logger.info(
            "TTM Squeeze Signals | Squeeze Fires: %d, Long: %d, Short: %d, Exits: %d",
            squeeze_fires,
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
