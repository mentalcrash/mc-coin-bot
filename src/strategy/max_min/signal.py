"""MAX/MIN Combined Signal Generator.

신고가 매수(trend) + 신저가 매수(mean reversion) 복합 시그널을 생성합니다.

Signal Formula:
    1. max_signal = (close > rolling_max)  # 신고가 돌파 → 매수 (trend)
    2. min_signal = (close < rolling_min)  # 신저가 돌파 → 매수 (MR)
    3. combined = max_weight * max_signal + min_weight * min_signal
    4. strength = combined * vol_scalar.shift(1)  # Shift(1) Rule
    5. direction = sign(strength)

Note:
    - rolling_max, rolling_min은 preprocessor에서 이미 shift(1) 적용됨
    - vol_scalar에는 signal.py에서 추가 shift(1) 적용
    - 두 시그널 모두 양수 (매수)이므로 Long-only by design
    - Short mode는 direction이 항상 >= 0이므로 실질적으로 영향 없음

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.max_min.config import MaxMinConfig
from src.strategy.tsmom.config import ShortMode
from src.strategy.types import Direction, StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: MaxMinConfig | None = None,
) -> StrategySignals:
    """MAX/MIN 복합 시그널 생성.

    전처리된 DataFrame에서 신고가/신저가 기반 복합 진입/청산 시그널을 계산합니다.

    Signal Generation Pipeline:
        1. 신고가/신저가 시그널 계산 (rolling_max/min은 이미 shift(1) 적용됨)
        2. 가중 결합: max_weight * max_signal + min_weight * min_signal
        3. Vol scalar shift(1) 적용 후 strength 계산
        4. Entry/Exit 시그널 생성

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: close, rolling_max, rolling_min, vol_scalar
        config: MAX/MIN 설정

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    if config is None:
        config = MaxMinConfig()

    # 입력 검증
    required_cols = {"close", "rolling_max", "rolling_min", "vol_scalar"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # ================================================================
    # 1. 컬럼 추출 (rolling_max/min은 preprocessor에서 이미 shift(1) 적용됨)
    # ================================================================
    close: pd.Series = df["close"]  # type: ignore[assignment]
    rolling_max: pd.Series = df["rolling_max"]  # type: ignore[assignment]
    rolling_min: pd.Series = df["rolling_min"]  # type: ignore[assignment]
    vol_scalar: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    # ================================================================
    # 2. MAX/MIN 시그널 계산
    # ================================================================
    # 신고가 돌파 → 매수 (trend-following)
    max_signal = (close > rolling_max).astype(float)

    # 신저가 돌파 → 매수 (mean-reversion: 저가 매수)
    min_signal = (close < rolling_min).astype(float)

    # 가중 결합
    combined: pd.Series = (  # type: ignore[assignment]
        config.max_weight * max_signal + config.min_weight * min_signal
    )

    # ================================================================
    # 3. Strength 계산 (vol_scalar에 shift(1) 적용)
    # ================================================================
    # vol_scalar는 preprocessor에서 shift되지 않았으므로 여기서 shift(1)
    vol_scalar_shifted: pd.Series = vol_scalar.shift(1)  # type: ignore[assignment]
    strength = pd.Series(
        (combined * vol_scalar_shifted).fillna(0),
        index=df.index,
        name="strength",
    )

    # ================================================================
    # 4. Direction 계산
    # ================================================================
    direction_raw = pd.Series(np.sign(strength), index=df.index)
    direction = pd.Series(
        direction_raw.fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    # ================================================================
    # 5. ShortMode 처리
    # ================================================================
    # 두 시그널 모두 양수(매수)이므로 Long-only by design
    # 하지만 패턴 일관성을 위해 ShortMode 처리 포함
    if config.short_mode == ShortMode.DISABLED:
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    # HEDGE_ONLY, FULL은 이 전략에서는 실질적으로 영향 없음
    # (combined >= 0 항상 성립)

    # ================================================================
    # 6. Entry/Exit 시그널 생성
    # ================================================================
    prev_direction = direction.shift(1).fillna(0)

    # Long 진입: direction이 1이 되는 순간
    long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)

    # Short 진입: direction이 -1이 되는 순간 (이 전략에서는 거의 발생하지 않음)
    short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)

    entries = pd.Series(
        long_entry | short_entry,
        index=df.index,
        name="entries",
    )

    # 청산: 포지션이 non-zero에서 0으로 변할 때 또는 방향 반전
    to_neutral = (direction == Direction.NEUTRAL) & (prev_direction != Direction.NEUTRAL)
    reversal = direction * prev_direction < 0
    exits = pd.Series(
        to_neutral | reversal,
        index=df.index,
        name="exits",
    )

    # ================================================================
    # 7. 시그널 통계 로깅
    # ================================================================
    valid_strength = strength[strength != 0]
    long_signals = strength[strength > 0]

    max_triggered = int((close > rolling_max).sum())
    min_triggered = int((close < rolling_min).sum())

    if len(valid_strength) > 0:
        logger.info(
            "MAX/MIN Signals | Total: %d, Long: %d (%.1f%%), MAX triggers: %d, MIN triggers: %d",
            len(valid_strength),
            len(long_signals),
            len(long_signals) / len(valid_strength) * 100,
            max_triggered,
            min_triggered,
        )
        logger.info(
            "Entry/Exit Events | Entries: %d, Exits: %d",
            int(entries.sum()),
            int(exits.sum()),
        )

    return StrategySignals(
        entries=entries,
        exits=exits,
        direction=direction,
        strength=strength,
    )
