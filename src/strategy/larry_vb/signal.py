"""Larry Williams Volatility Breakout Signal Generator.

전일 변동폭 기반 돌파 시그널을 생성합니다.
1-bar hold: 돌파 발생 bar의 다음 bar에서만 포지션 유지.

Signal Logic:
    1. long_breakout = Close > breakout_upper (당일)
    2. short_breakout = Close < breakout_lower (당일)
    3. direction_raw = where(long, 1, where(short, -1, 0))
    4. direction = direction_raw.shift(1) → 1-bar hold
    5. strength = direction * vol_scalar

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.larry_vb.config import ShortMode
from src.strategy.types import Direction, StrategySignals

if TYPE_CHECKING:
    from src.strategy.larry_vb.config import LarryVBConfig


def generate_signals(
    df: pd.DataFrame,
    config: LarryVBConfig | None = None,
) -> StrategySignals:
    """Larry Williams Volatility Breakout 시그널 생성.

    전처리된 DataFrame에서 변동폭 돌파 시그널과 강도를 계산합니다.
    Shift(1) Rule을 적용하여 미래 참조 편향을 방지합니다.

    Signal Generation Pipeline:
        1. 돌파 판정: Close vs breakout_upper/lower (당일)
        2. direction_raw: long(+1), short(-1), neutral(0)
        3. Shift(1): 1-bar hold → 돌파 다음 바에서만 포지션
        4. Vol scalar로 strength 계산
        5. ShortMode 처리
        6. Entry/Exit 시그널 생성

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: close, breakout_upper, breakout_lower, vol_scalar
        config: Larry VB 설정

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    if config is None:
        from src.strategy.larry_vb.config import LarryVBConfig

        config = LarryVBConfig()

    # 입력 검증
    required_cols = {"close", "breakout_upper", "breakout_lower", "vol_scalar"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # ================================================================
    # 1. 돌파 판정 (당일 Close vs 당일 돌파 레벨)
    # ================================================================
    # breakout_upper = Open + k * prev_range (당일 시가 + 전일 변동폭)
    # 장중에 알 수 있는 값이므로 lookahead bias 없음
    close_series: pd.Series = df["close"]  # type: ignore[assignment]
    upper: pd.Series = df["breakout_upper"]  # type: ignore[assignment]
    lower: pd.Series = df["breakout_lower"]  # type: ignore[assignment]
    vol_scalar: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    long_breakout = close_series > upper
    short_breakout = close_series < lower

    # ================================================================
    # 2. Direction Raw (당일 판정)
    # ================================================================
    direction_raw = np.where(
        long_breakout,
        1,
        np.where(short_breakout, -1, 0),
    )

    # ================================================================
    # 3. Shift(1) 적용 — 1-bar hold
    # ================================================================
    # 돌파가 발생한 바의 다음 바에서만 포지션 보유
    # 1-bar hold이므로 ffill 없이 shift만 적용
    direction = pd.Series(direction_raw, index=df.index).shift(1).fillna(0).astype(int)

    # ================================================================
    # 4. Strength 계산 (direction * vol_scalar)
    # ================================================================
    # vol_scalar는 preprocessor에서 이미 shift(1) 적용됨
    strength = pd.Series(
        direction.astype(float) * vol_scalar.fillna(0),
        index=df.index,
        name="strength",
    )

    # ================================================================
    # 5. ShortMode 처리
    # ================================================================
    if config.short_mode == ShortMode.DISABLED:
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    # ================================================================
    # 6. Entry/Exit 시그널 생성
    # ================================================================
    prev_direction = direction.shift(1).fillna(0).astype(int)

    long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)
    short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)
    entries = pd.Series(long_entry | short_entry, index=df.index, name="entries")

    to_neutral = (direction == Direction.NEUTRAL) & (prev_direction != Direction.NEUTRAL)
    reversal = direction * prev_direction < 0
    exits = pd.Series(to_neutral | reversal, index=df.index, name="exits")

    # NaN 처리
    entries = entries.fillna(False)
    exits = exits.fillna(False)
    direction = pd.Series(direction.fillna(0).astype(int), index=df.index, name="direction")
    strength = pd.Series(strength.fillna(0.0), index=df.index, name="strength")

    # 시그널 통계 로깅
    long_entries = int(long_entry.sum())
    short_entries = int(short_entry.sum())
    total_exits = int(exits.sum())

    if long_entries > 0 or short_entries > 0:
        logger.info(
            "Larry VB Signals | Long: %d, Short: %d, Exits: %d",
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
