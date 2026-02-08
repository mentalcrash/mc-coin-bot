"""Vol-Regime Adaptive Signal Generator.

이 모듈은 전처리된 데이터에서 매매 시그널을 생성합니다.
VectorBT 및 QuantStats와 호환되는 표준 출력을 제공합니다.

Signal Formula:
    1. regime_strength = shift(1) for lookahead prevention
    2. direction = sign(regime_strength)
    3. strength = regime_strength (vol scaling 포함)

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.types import Direction, StrategySignals
from src.strategy.vol_regime.config import ShortMode, VolRegimeConfig


def generate_signals(
    df: pd.DataFrame,
    config: VolRegimeConfig | None = None,
) -> StrategySignals:
    """Vol-Regime Adaptive 시그널 생성.

    전처리된 DataFrame에서 진입/청산 시그널과 강도를 계산합니다.
    Shift(1) Rule을 적용하여 미래 참조 편향을 방지합니다.

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: regime_strength
        config: Vol-Regime 설정. None이면 기본 설정 사용.

    Returns:
        StrategySignals NamedTuple:
            - entries: 진입 시그널 (bool Series)
            - exits: 청산 시그널 (bool Series)
            - direction: 방향 시리즈 (-1, 0, 1)
            - strength: 시그널 강도 (레버리지 무제한)

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    if config is None:
        config = VolRegimeConfig()

    # 입력 검증
    required_cols = {"regime_strength"}

    # HEDGE_ONLY 모드에서는 drawdown 컬럼 필요
    if config.short_mode == ShortMode.HEDGE_ONLY:
        required_cols.add("drawdown")

    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 1. regime_strength에서 시그널 추출 (이미 vol_scalar 포함)
    regime_strength_series: pd.Series = df["regime_strength"]  # type: ignore[assignment]

    # 2. Shift(1) 적용: 전봉 기준 시그널 (미래 참조 편향 방지)
    signal_shifted: pd.Series = regime_strength_series.shift(1)  # type: ignore[assignment]

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

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        # 헤지 모드: 드로다운 임계값 초과 시에만 숏 허용
        drawdown_series: pd.Series = df["drawdown"]  # type: ignore[assignment]
        hedge_active = drawdown_series < config.hedge_threshold

        # 헤지 비활성 시 숏 -> 중립
        short_mask = direction == Direction.SHORT
        suppress_short = short_mask & ~hedge_active
        direction = direction.where(~suppress_short, Direction.NEUTRAL)
        strength = strength.where(~suppress_short, 0.0)

        # 헤지 활성 시 숏 강도 조절
        active_short = short_mask & hedge_active
        strength = strength.where(
            ~active_short,
            strength * config.hedge_strength_ratio,
        )

        # 헤지 활성화 통계 로깅
        hedge_days = int(hedge_active.sum())
        if hedge_days > 0:
            logger.info(
                "Hedge Mode | Active: {} days ({:.1f}%), Threshold: {:.1f}%",
                hedge_days,
                hedge_days / len(hedge_active) * 100,
                config.hedge_threshold * 100,
            )

    # else: ShortMode.FULL - 모든 시그널 그대로 유지

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
