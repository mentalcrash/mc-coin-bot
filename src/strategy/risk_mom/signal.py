"""Risk-Managed Momentum Signal Generator.

TSMOM과 동일한 시그널 구조를 사용하되, vol_scalar 대신
BSC variance scaling으로 포지션 크기를 결정합니다.

Signal Formula:
    1. momentum_direction = sign(vw_momentum)
    2. scaled_momentum = momentum_direction * bsc_scaling
    3. Shift(1) 적용 — 미래 참조 편향 방지
    4. direction = sign(shifted_signal)
    5. strength = shifted_signal

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
from src.strategy.types import Direction

if TYPE_CHECKING:
    from src.strategy.risk_mom.config import RiskMomConfig
    from src.strategy.types import StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: RiskMomConfig | None = None,
) -> StrategySignals:
    """Risk-Managed Momentum 시그널 생성.

    BSC variance scaling을 적용한 모멘텀 시그널을 생성합니다.
    Shift(1) Rule을 적용하여 미래 참조 편향을 방지합니다.

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: vw_momentum, bsc_scaling
        config: Risk-Mom 설정. None이면 기본 설정 사용.

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    from src.strategy.risk_mom.config import RiskMomConfig
    from src.strategy.types import StrategySignals

    if config is None:
        config = RiskMomConfig()

    # 입력 검증
    required_cols = {"vw_momentum", "bsc_scaling"}

    if config.short_mode == ShortMode.HEDGE_ONLY:
        required_cols.add("drawdown")

    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 1. BSC Scaled Momentum 계산
    momentum_series: pd.Series = df["vw_momentum"]  # type: ignore[assignment]
    bsc_scaling_series: pd.Series = df["bsc_scaling"]  # type: ignore[assignment]

    # 모멘텀 방향 * BSC scaling
    momentum_direction = np.sign(momentum_series)
    scaled_momentum = momentum_direction * bsc_scaling_series

    # 2. Shift(1) 적용 — 전봉 기준 시그널
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

    # 5. 숏 모드 처리
    if config.short_mode == ShortMode.DISABLED:
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        drawdown_series: pd.Series = df["drawdown"]  # type: ignore[assignment]
        hedge_active = drawdown_series < config.hedge_threshold

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
                "Risk-Mom Hedge | Active: {} days ({:.1f}%), Threshold: {:.1f}%",
                hedge_days,
                hedge_days / len(hedge_active) * 100,
                config.hedge_threshold * 100,
            )

    # 6. Entry/Exit 시그널 생성
    prev_direction = direction.shift(1).fillna(0)

    long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)
    short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)
    entries = pd.Series(
        long_entry | short_entry,
        index=df.index,
        name="entries",
    )

    to_neutral = (direction == Direction.NEUTRAL) & (prev_direction != Direction.NEUTRAL)
    reversal = direction * prev_direction < 0
    exits = pd.Series(
        to_neutral | reversal,
        index=df.index,
        name="exits",
    )

    # 디버그 로깅
    valid_strength = strength[strength != 0]
    long_signals = strength[strength > 0]
    short_signals = strength[strength < 0]

    if len(valid_strength) > 0:
        logger.info(
            "Risk-Mom Signals | Total: {}, Long: {} ({:.1f}%), Short: {} ({:.1f}%)",
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
