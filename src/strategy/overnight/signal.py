"""Overnight Seasonality Signal Generator.

시간대 기반 진입/청산 시그널을 생성합니다.
entry_hour에 진입, exit_hour에 청산하며 자정 넘김(wrap-around)을 지원합니다.

Signal Formula:
    1. in_position = (hour >= entry_hour) | (hour < exit_hour)  [wrap-around]
    2. in_position_shifted = in_position.shift(1)  [Shift(1) Rule]
    3. direction = in_position_shifted (1 = Long, 0 = Neutral)
    4. strength = direction * vol_scalar.shift(1)

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from src.strategy.tsmom.config import ShortMode
from src.strategy.types import Direction, StrategySignals

if TYPE_CHECKING:
    from src.strategy.overnight.config import OvernightConfig


def generate_signals(
    df: pd.DataFrame,
    config: OvernightConfig | None = None,
) -> StrategySignals:
    """Overnight Seasonality 시그널 생성.

    전처리된 DataFrame에서 시간대 기반 진입/청산 시그널과 강도를 계산합니다.
    Shift(1) Rule을 적용하여 미래 참조 편향을 방지합니다.

    Signal Generation Pipeline:
        1. 시간대 기반 포지션 판단 (entry_hour ~ exit_hour)
        2. Shift(1) 적용 (미래 참조 편향 방지)
        3. Direction 생성 (Long/Neutral)
        4. Vol scalar로 strength 계산
        5. Vol filter 적용 (선택적)
        6. ShortMode 처리 (기본 Long-Only)
        7. Entry/Exit 시그널 생성

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: hour, vol_scalar
        config: Overnight 설정

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    if config is None:
        from src.strategy.overnight.config import OvernightConfig

        config = OvernightConfig()

    # 입력 검증
    required_cols = {"hour", "vol_scalar"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # ================================================================
    # 1. 시간대 기반 포지션 판단
    # ================================================================
    hour: pd.Series = df["hour"]  # type: ignore[assignment]
    entry_hour = config.entry_hour
    exit_hour = config.exit_hour

    # Handle wrap-around case: entry=22, exit=0 means 22:00-23:59 (2 hours)
    if entry_hour > exit_hour:
        # Wrap around midnight: hour >= 22 OR hour < 0
        in_position = (hour >= entry_hour) | (hour < exit_hour)
    else:
        # Same day: entry <= hour < exit (e.g., entry=9, exit=17)
        in_position = (hour >= entry_hour) & (hour < exit_hour)

    # ================================================================
    # 2. Shift(1) 적용 — 미래 참조 편향 방지
    # ================================================================
    in_position_shifted = pd.Series(
        in_position.shift(1).fillna(0).astype(bool),
        index=df.index,
    )

    # ================================================================
    # 3. Direction 계산
    # ================================================================
    direction = pd.Series(
        in_position_shifted.astype(int),
        index=df.index,
        name="direction",
    )

    # ================================================================
    # 4. Strength 계산 (direction * vol_scalar)
    # ================================================================
    vol_scalar: pd.Series = df["vol_scalar"]  # type: ignore[assignment]
    vol_scalar_shifted: pd.Series = vol_scalar.shift(1)  # type: ignore[assignment]

    raw_strength: pd.Series = direction * vol_scalar_shifted  # type: ignore[assignment]
    strength = pd.Series(
        raw_strength.fillna(0),
        index=df.index,
        name="strength",
    )

    # ================================================================
    # 5. Vol filter: 고변동성 구간에서 strength 스케일업
    # ================================================================
    if config.use_vol_filter and "rolling_vol_ratio" in df.columns:
        vol_ratio: pd.Series = df["rolling_vol_ratio"]  # type: ignore[assignment]
        vol_ratio_shifted: pd.Series = vol_ratio.shift(1)  # type: ignore[assignment]
        high_vol_mask = vol_ratio_shifted > config.vol_filter_threshold
        strength = strength.where(~high_vol_mask, strength * 1.5)

    # ================================================================
    # 6. ShortMode 처리 (기본 Long-Only이므로 대부분 불필요)
    # ================================================================
    if config.short_mode == ShortMode.DISABLED:
        # Long-Only: 숏 시그널을 중립으로 변환
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    # ================================================================
    # 7. Entry/Exit 시그널 생성
    # ================================================================
    prev_direction = direction.shift(1).fillna(0)

    # Long 진입: direction이 1이 되는 순간
    long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)

    # Short 진입: direction이 -1이 되는 순간
    short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)

    entries = pd.Series(
        long_entry | short_entry,
        index=df.index,
        name="entries",
    )

    # 청산: 포지션이 non-zero에서 0으로 변할 때, 또는 방향 반전
    to_neutral = (direction == Direction.NEUTRAL) & (prev_direction != Direction.NEUTRAL)
    reversal = direction * prev_direction < 0
    exits = pd.Series(
        to_neutral | reversal,
        index=df.index,
        name="exits",
    )

    # 시그널 통계 로깅
    valid_strength = strength[strength != 0]
    entry_count = int(entries.sum())
    exit_count = int(exits.sum())

    if len(valid_strength) > 0:
        in_position_pct = len(valid_strength) / len(strength) * 100
        logger.info(
            "Overnight Signals | In-Position: %.1f%%, Entries: %d, Exits: %d, Entry: %02d:00, Exit: %02d:00 UTC",
            in_position_pct,
            entry_count,
            exit_count,
            config.entry_hour,
            config.exit_hour,
        )

    return StrategySignals(
        entries=entries,
        exits=exits,
        direction=direction,
        strength=strength,
    )
