"""Multi-Factor Ensemble Signal Generator.

이 모듈은 전처리된 데이터에서 매매 시그널을 생성합니다.
3개 팩터의 결합 점수(combined_score)와 변동성 스케일러를 사용하여
포지션 방향과 강도를 결정합니다.

Signal Formula:
    1. direction = sign(combined_score) (shift(1) 적용)
    2. strength = direction * vol_scalar (shift(1) 적용)

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

from src.strategy.multi_factor.config import MultiFactorConfig
from src.strategy.tsmom.config import ShortMode
from src.strategy.types import Direction

if TYPE_CHECKING:
    from src.strategy.types import StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: MultiFactorConfig | None = None,
) -> StrategySignals:
    """Multi-Factor Ensemble 시그널 생성.

    전처리된 DataFrame에서 진입/청산 시그널과 강도를 계산합니다.
    Shift(1) Rule을 적용하여 미래 참조 편향을 방지합니다.

    Signal Generation Pipeline:
        1. combined_score의 방향 추출 (sign)
        2. vol_scalar로 강도 스케일링
        3. Shift(1) 적용 (미래 참조 편향 방지)
        4. ShortMode에 따른 시그널 필터링
        5. Entry/Exit 시그널 생성

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: combined_score, vol_scalar
        config: Multi-Factor 설정. None이면 기본 설정 사용.

    Returns:
        StrategySignals NamedTuple:
            - entries: 진입 시그널 (bool Series)
            - exits: 청산 시그널 (bool Series)
            - direction: 방향 시리즈 (-1, 0, 1)
            - strength: 시그널 강도 (레버리지 무제한)

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    # Avoid circular import by importing here
    from src.strategy.types import StrategySignals

    if config is None:
        config = MultiFactorConfig()

    # 입력 검증
    required_cols = {"combined_score", "vol_scalar"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 1. Combined Score → 방향 + 강도
    combined_score: pd.Series = df["combined_score"]  # type: ignore[assignment]
    vol_scalar_series: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    # direction * vol_scalar
    score_direction = np.sign(combined_score)
    scaled_signal = score_direction * vol_scalar_series

    # 2. Shift(1) 적용: 전봉 기준 시그널 (미래 참조 편향 방지)
    signal_shifted: pd.Series = scaled_signal.shift(1)  # type: ignore[assignment]

    # 3. Direction 계산
    direction_raw = pd.Series(np.sign(signal_shifted), index=df.index)
    direction = pd.Series(
        direction_raw.fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    # 4. 강도 계산
    strength = pd.Series(
        signal_shifted.fillna(0),
        index=df.index,
        name="strength",
    )

    # 5. 숏 모드에 따른 시그널 처리
    if config.short_mode == ShortMode.DISABLED:
        # Long-Only: 모든 숏 시그널을 중립으로 변환
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    # else: ShortMode.FULL - 모든 시그널 그대로 유지
    # HEDGE_ONLY는 이 전략에서 미지원 (drawdown 컬럼 미계산)

    # 6. 진입 시그널: 포지션이 0에서 non-zero로 변할 때
    prev_direction = direction.shift(1).fillna(0)

    # Long 진입
    long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)
    # Short 진입
    short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)

    entries = pd.Series(
        long_entry | short_entry,
        index=df.index,
        name="entries",
    )

    # 7. 청산 시그널: 포지션이 non-zero에서 0으로 변할 때 또는 방향 반전
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
