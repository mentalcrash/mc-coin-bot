"""Enhanced VW-TSMOM Signal Generator.

볼륨 비율 정규화 기반 모멘텀 시그널을 생성합니다.
기존 TSMOM signal.py와 동일한 패턴을 따르되, evw_momentum 컬럼을 사용합니다.

Signal Formula:
    1. scaled_momentum = sign(evw_momentum) * vol_scalar
    2. strength = scaled_momentum.shift(1)
    3. ShortMode 처리
    4. Entry/Exit 시그널 생성

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.enhanced_tsmom.config import EnhancedTSMOMConfig, ShortMode
from src.strategy.types import Direction, StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: EnhancedTSMOMConfig | None = None,
) -> StrategySignals:
    """Enhanced VW-TSMOM 시그널 생성.

    전처리된 DataFrame에서 진입/청산 시그널과 강도를 계산합니다.
    Shift(1) Rule을 적용하여 미래 참조 편향을 방지합니다.

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: evw_momentum, vol_scalar
        config: Enhanced TSMOM 설정

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    if config is None:
        config = EnhancedTSMOMConfig()

    # 입력 검증
    required_cols = {"evw_momentum", "vol_scalar"}

    # HEDGE_ONLY 모드에서는 drawdown 컬럼 필요
    if config.short_mode == ShortMode.HEDGE_ONLY:
        required_cols.add("drawdown")

    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 1. Scaled Momentum 계산
    momentum_series: pd.Series = df["evw_momentum"]  # type: ignore[assignment]
    vol_scalar_series: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    momentum_direction = np.sign(momentum_series)
    scaled_momentum = momentum_direction * vol_scalar_series

    # 2. Shift(1) 적용 (미래 참조 편향 방지)
    signal_shifted: pd.Series = scaled_momentum.shift(1)  # type: ignore[assignment]

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

        hedge_days = int(hedge_active.sum())
        if hedge_days > 0:
            logger.info(
                "Hedge Mode | Active: %d days (%.1f%%), Threshold: %.1f%%",
                hedge_days,
                hedge_days / len(hedge_active) * 100,
                config.hedge_threshold * 100,
            )

    # else: ShortMode.FULL - 모든 시그널 그대로 유지

    # 6. 진입 시그널
    prev_direction = direction.shift(1).fillna(0)

    long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)
    short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)
    entries = pd.Series(
        long_entry | short_entry,
        index=df.index,
        name="entries",
    )

    # 7. 청산 시그널
    to_neutral = (direction == Direction.NEUTRAL) & (prev_direction != Direction.NEUTRAL)
    reversal = direction * prev_direction < 0
    exits = pd.Series(
        to_neutral | reversal,
        index=df.index,
        name="exits",
    )

    # 시그널 통계 로깅
    valid_strength = strength[strength != 0]
    long_signals = strength[strength > 0]
    short_signals = strength[strength < 0]

    if len(valid_strength) > 0:
        logger.info(
            "Enhanced VW-TSMOM Signals | Total: %d, Long: %d (%.1f%%), Short: %d (%.1f%%)",
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
