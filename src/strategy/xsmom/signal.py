"""XSMOM Signal Generator.

이 모듈은 전처리된 데이터에서 매매 시그널을 생성합니다.
VectorBT 및 QuantStats와 호환되는 표준 출력을 제공합니다.

Signal Formula:
    1. raw_signal = sign(rolling_return) * vol_scalar
    2. held_signal = holding_period 필터 적용
    3. signal = shift(1) 적용 (미래 참조 편향 방지)
    4. direction = sign(signal)
    5. strength = signal (변동성 스케일링된 시그널 강도)

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

from src.strategy.tsmom.config import ShortMode
from src.strategy.xsmom.config import XSMOMConfig
from src.strategy.xsmom.preprocessor import calculate_holding_signal

if TYPE_CHECKING:
    from src.strategy.types import StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: XSMOMConfig | None = None,
) -> StrategySignals:
    """XSMOM 시그널 생성.

    전처리된 DataFrame에서 진입/청산 시그널과 강도를 계산합니다.
    Shift(1) Rule을 적용하여 미래 참조 편향을 방지합니다.

    Signal Generation Pipeline:
        1. raw_signal = sign(rolling_return) * vol_scalar
        2. holding_period 필터 적용
        3. Shift(1) 적용
        4. ShortMode에 따른 필터링
        5. Entry/Exit 시그널 생성

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: rolling_return, vol_scalar
        config: XSMOM 설정 (None이면 기본 설정 사용)

    Returns:
        StrategySignals NamedTuple:
            - entries: 진입 시그널 (bool Series)
            - exits: 청산 시그널 (bool Series)
            - direction: 방향 시리즈 (-1, 0, 1)
            - strength: 시그널 강도

    Raises:
        ValueError: 필수 컬럼 누락 시

    Example:
        >>> from src.strategy.xsmom.preprocessor import preprocess
        >>> processed_df = preprocess(ohlcv_df, config)
        >>> signals = generate_signals(processed_df, config)
    """
    from src.strategy.types import Direction, StrategySignals

    # 기본 config 설정
    if config is None:
        config = XSMOMConfig()

    # 입력 검증
    required_cols = {"rolling_return", "vol_scalar"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 1. Raw signal 계산: sign(rolling_return) * vol_scalar
    rolling_return_series: pd.Series = df["rolling_return"]  # type: ignore[assignment]
    vol_scalar_series: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    return_direction = pd.Series(np.sign(rolling_return_series), index=df.index)
    raw_signal: pd.Series = return_direction * vol_scalar_series  # type: ignore[assignment]

    # 2. Holding period 필터 적용
    held_signal = calculate_holding_signal(raw_signal, config.holding_period)

    # 3. Shift(1) 적용: 전봉 기준 시그널 (미래 참조 편향 방지)
    signal_shifted: pd.Series = held_signal.shift(1)  # type: ignore[assignment]

    # 4. Direction 계산
    direction_raw = pd.Series(np.sign(signal_shifted), index=df.index)
    direction = pd.Series(
        direction_raw.fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    # 5. Strength 계산
    strength = pd.Series(
        signal_shifted.fillna(0),
        index=df.index,
        name="strength",
    )

    # 6. ShortMode에 따른 시그널 처리
    if config.short_mode == ShortMode.DISABLED:
        # Long-Only: 모든 숏 시그널을 중립으로 변환
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    # ShortMode.FULL: 모든 시그널 그대로 유지
    # ShortMode.HEDGE_ONLY: XSMOM에서는 FULL과 동일하게 처리
    # (헤지 로직은 PM 레벨에서 처리)

    # 7. 진입 시그널: 포지션이 변할 때
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

    # 8. 청산 시그널: 포지션이 청산되거나 방향이 반전될 때
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
            "XSMOM Signal Stats | Total: %d signals, Long: %d (%.1f%%), Short: %d (%.1f%%)",
            len(valid_strength),
            len(long_signals),
            len(long_signals) / len(valid_strength) * 100,
            len(short_signals),
            len(short_signals) / len(valid_strength) * 100,
        )
        logger.info(
            "XSMOM Entry/Exit | Long entries: %d, Short entries: %d, Exits: %d, Reversals: %d",
            int(long_entry.sum()),
            int(short_entry.sum()),
            int(exits.sum()),
            int(reversal.sum()),
        )

    return StrategySignals(
        entries=entries,
        exits=exits,
        direction=direction,
        strength=strength,
    )
