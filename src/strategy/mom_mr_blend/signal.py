"""Momentum + Mean Reversion Blend Signal Generator.

모멘텀 z-score와 평균회귀 z-score를 블렌딩하여 매매 시그널을 생성합니다.
모멘텀은 추세 추종(sign), 평균회귀는 역추세(-sign)로 동작합니다.

Signal Formula:
    1. prev_mom_z = mom_zscore.shift(1)
    2. prev_mr_z = mr_zscore.shift(1)
    3. mom_signal = sign(prev_mom_z)
    4. mr_signal = -sign(prev_mr_z)
    5. combined = mom_weight * mom_signal + mr_weight * mr_signal
    6. direction = sign(combined)
    7. strength = combined * vol_scalar

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.mom_mr_blend.config import ShortMode
from src.strategy.types import Direction, StrategySignals

if TYPE_CHECKING:
    from src.strategy.mom_mr_blend.config import MomMrBlendConfig

logger = logging.getLogger(__name__)


def generate_signals(
    df: pd.DataFrame,
    config: MomMrBlendConfig | None = None,
) -> StrategySignals:
    """Momentum + Mean Reversion Blend 시그널 생성.

    전처리된 DataFrame에서 블렌드 시그널을 계산합니다.

    Signal Generation Pipeline:
        1. Z-score shift(1) (미래 참조 편향 방지)
        2. 모멘텀 sign + 평균회귀 contrarian sign 블렌딩
        3. Vol scalar 적용 (변동성 기반 포지션 사이징)
        4. ShortMode 처리
        5. Entry/Exit 시그널 생성

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: mom_zscore, mr_zscore, vol_scalar
        config: Mom-MR Blend 설정

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    if config is None:
        from src.strategy.mom_mr_blend.config import MomMrBlendConfig

        config = MomMrBlendConfig()

    # 입력 검증
    required_cols = {"mom_zscore", "mr_zscore", "vol_scalar"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # ================================================================
    # 1. Z-Score Shift(1) - 미래 참조 편향 방지
    # ================================================================
    prev_mom_z: pd.Series = df["mom_zscore"].shift(1)  # type: ignore[assignment]
    prev_mr_z: pd.Series = df["mr_zscore"].shift(1)  # type: ignore[assignment]
    vol_scalar: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    # ================================================================
    # 2. 개별 시그널 계산
    # ================================================================
    # 모멘텀: z-score 부호 추종
    mom_signal = pd.Series(np.sign(prev_mom_z), index=df.index)
    # 평균회귀: z-score 부호 반대 (contrarian)
    mr_signal = pd.Series(-np.sign(prev_mr_z), index=df.index)

    # ================================================================
    # 3. 블렌딩
    # ================================================================
    combined = config.mom_weight * mom_signal + config.mr_weight * mr_signal

    # ================================================================
    # 4. Direction 판정
    # ================================================================
    direction = pd.Series(
        np.sign(combined).fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    # ================================================================
    # 5. Strength = combined * vol_scalar
    # ================================================================
    strength = pd.Series(
        combined.fillna(0) * vol_scalar.fillna(0),
        index=df.index,
        name="strength",
    )

    # ================================================================
    # 6. ShortMode 처리
    # ================================================================
    if config.short_mode == ShortMode.DISABLED:
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        # HEDGE_ONLY는 추가 drawdown 데이터가 필요하나,
        # 이 전략에서는 DISABLED/FULL만 사용 권장
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    # ================================================================
    # 7. Entry/Exit 시그널 생성
    # ================================================================
    prev_direction = direction.shift(1).fillna(0)

    long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)
    short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)
    entries = pd.Series(long_entry | short_entry, index=df.index, name="entries")

    to_neutral = (direction == Direction.NEUTRAL) & (prev_direction != Direction.NEUTRAL)
    reversal = direction * prev_direction < 0
    exits = pd.Series(to_neutral | reversal, index=df.index, name="exits")

    # 시그널 통계 로깅
    valid_strength = strength[strength != 0]
    long_signals = strength[strength > 0]
    short_signals = strength[strength < 0]

    if len(valid_strength) > 0:
        logger.info(
            "Mom-MR Blend Signals | Total: %d, Long: %d (%.1f%%), Short: %d (%.1f%%)",
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
