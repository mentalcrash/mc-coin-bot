"""HAR Volatility Overlay Signal Generator.

이 모듈은 전처리된 데이터에서 매매 시그널을 생성합니다.
VectorBT 및 QuantStats와 호환되는 표준 출력을 제공합니다.

Signal Formula:
    1. vol_surprise > threshold → momentum (follow recent returns) with vol_scalar
    2. vol_surprise < -threshold → mean-reversion (-recent returns) with vol_scalar * 0.5
    3. |vol_surprise| <= threshold → neutral
    4. Shift(1) 적용: 미래 참조 편향 방지

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

from src.strategy.har_vol.config import HARVolConfig
from src.strategy.tsmom.config import ShortMode
from src.strategy.types import Direction

if TYPE_CHECKING:
    from src.strategy.types import StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: HARVolConfig | None = None,
) -> StrategySignals:
    """HAR Volatility Overlay 시그널 생성.

    전처리된 DataFrame에서 진입/청산 시그널과 강도를 계산합니다.
    Shift(1) Rule을 적용하여 미래 참조 편향을 방지합니다.

    Signal Logic:
        - vol_surprise > threshold → momentum direction (sign of recent returns)
          with vol_scalar sizing
        - vol_surprise < -threshold → mean-reversion direction (-sign of recent returns)
          with vol_scalar * 0.5 sizing (conservative)
        - |vol_surprise| <= threshold → neutral (no position)

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: vol_surprise, returns, vol_scalar
        config: HAR Volatility 설정. None이면 기본 설정 사용.

    Returns:
        StrategySignals NamedTuple:
            - entries: 진입 시그널 (bool Series)
            - exits: 청산 시그널 (bool Series)
            - direction: 방향 시리즈 (-1, 0, 1)
            - strength: 시그널 강도 (레버리지 무제한)

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    from src.strategy.types import StrategySignals

    if config is None:
        config = HARVolConfig()

    # 입력 검증
    required_cols = {"vol_surprise", "returns", "vol_scalar"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 1. Shift(1) 적용: 전봉 기준 시그널 (미래 참조 편향 방지)
    vol_surprise_prev: pd.Series = df["vol_surprise"].shift(1)  # type: ignore[assignment]
    returns_prev: pd.Series = df["returns"].shift(1)  # type: ignore[assignment]
    vol_scalar_prev: pd.Series = df["vol_scalar"].shift(1)  # type: ignore[assignment]

    threshold = config.vol_surprise_threshold

    # 2. Vol surprise 방향 분류
    # Positive surprise → momentum (follow direction of recent returns)
    # Negative surprise → mean-reversion (opposite of recent returns)
    momentum_mask = vol_surprise_prev > threshold
    mean_reversion_mask = vol_surprise_prev < -threshold

    # 3. Direction 계산
    # Momentum: follow recent returns sign
    # Mean-reversion: oppose recent returns sign, scaled to 0.5
    returns_sign = np.sign(returns_prev)

    direction_raw = pd.Series(
        np.where(
            momentum_mask,
            returns_sign,  # momentum: follow returns direction
            np.where(
                mean_reversion_mask,
                -returns_sign,  # mean-reversion: oppose returns direction
                0,  # neutral: no position
            ),
        ),
        index=df.index,
    )

    # 4. Strength = direction * vol_scalar (momentum) or direction * vol_scalar * 0.5 (MR)
    strength_raw = pd.Series(
        np.where(
            momentum_mask,
            direction_raw * vol_scalar_prev,
            np.where(
                mean_reversion_mask,
                direction_raw * vol_scalar_prev * 0.5,
                0.0,
            ),
        ),
        index=df.index,
    )

    # 5. Direction 정규화
    direction = pd.Series(
        pd.Series(np.sign(strength_raw), index=df.index).fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    # 6. 강도 계산
    strength = pd.Series(
        strength_raw.fillna(0),
        index=df.index,
        name="strength",
    )

    # 7. 숏 모드에 따른 시그널 처리
    if config.short_mode == ShortMode.DISABLED:
        # Long-Only: 모든 숏 시그널을 중립으로 변환
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    # else: ShortMode.FULL - 모든 시그널 그대로 유지
    # NOTE: HEDGE_ONLY는 HAR-Vol에서 미지원 (drawdown 기반 헤지 불필요)

    # 8. 진입 시그널: 포지션이 0에서 non-zero로 변할 때
    prev_direction = direction.shift(1).fillna(0)

    long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)
    short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)

    entries = pd.Series(
        long_entry | short_entry,
        index=df.index,
        name="entries",
    )

    # 9. 청산 시그널: 포지션이 non-zero에서 0으로 변할 때 또는 방향 반전
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
        msg = (
            "HAR-Vol Signal Statistics | Total: {} signals, Long: {} ({:.1f}%), Short: {} ({:.1f}%)"
        )
        logger.info(
            msg,
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
