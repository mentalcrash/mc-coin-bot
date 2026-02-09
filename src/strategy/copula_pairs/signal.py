"""Copula Pairs Trading Signal Generator.

Spread z-score 기반 평균 회귀 시그널을 생성합니다.
Stateful 시그널: 포지션은 exit 또는 stop 조건까지 유지됩니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (state machine via ffill)
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.copula_pairs.config import CopulaPairsConfig
from src.strategy.tsmom.config import ShortMode
from src.strategy.types import Direction

if TYPE_CHECKING:
    from src.strategy.types import StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: CopulaPairsConfig | None = None,
) -> StrategySignals:
    """Copula Pairs 시그널 생성 (Spread Z-score 기반 Mean-Reversion).

    전처리된 DataFrame에서 진입/청산 시그널과 강도를 계산합니다.
    Shift(1) Rule을 적용하여 미래 참조 편향을 방지합니다.

    Signal Logic:
        - spread_zscore >= zscore_entry -> SHORT spread (short coin, long pair)
          -> direction = -1, strength = vol_scalar
        - spread_zscore <= -zscore_entry -> LONG spread (long coin, short pair)
          -> direction = 1, strength = vol_scalar
        - |spread_zscore| <= zscore_exit -> EXIT (close position)
          -> direction = 0
        - |spread_zscore| >= zscore_stop -> STOP (emergency close)
          -> direction = 0

    Important:
        - This is a STATEFUL signal (positions held until exit/stop).
        - Uses a vectorized state machine approach (ffill).
        - shift(1) applied to prevent lookahead bias.

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: spread_zscore, vol_scalar
        config: CopulaPairs 설정. None이면 기본 설정 사용.

    Returns:
        StrategySignals NamedTuple (entries, exits, direction, strength)

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    from src.strategy.types import StrategySignals

    # 기본 config 설정
    if config is None:
        config = CopulaPairsConfig()

    # 입력 검증
    required_cols = {"spread_zscore", "vol_scalar"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 컬럼 추출
    spread_zscore: pd.Series = df["spread_zscore"]  # type: ignore[assignment]
    vol_scalar_series: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    # 1. Build position state (vectorized state machine)
    # Entry signals
    long_entry = spread_zscore <= -config.zscore_entry
    short_entry = spread_zscore >= config.zscore_entry
    exit_signal = spread_zscore.abs() <= config.zscore_exit
    stop_signal = spread_zscore.abs() >= config.zscore_stop

    # Build raw position: 1 for long, -1 for short, 0 for exit/stop, NaN for hold
    raw_position = pd.Series(np.nan, index=df.index)
    raw_position[long_entry] = 1
    raw_position[short_entry] = -1
    raw_position[exit_signal | stop_signal] = 0

    # Forward fill to maintain position (stateful)
    position = raw_position.ffill().fillna(0)

    # 2. Shift(1) 적용: 전봉 기준 시그널 (미래 참조 편향 방지)
    position_shifted: pd.Series = position.shift(1).fillna(0)  # type: ignore[assignment]
    vol_scalar_shifted: pd.Series = vol_scalar_series.shift(1).fillna(0)  # type: ignore[assignment]

    # 3. Direction 계산
    direction = pd.Series(
        position_shifted.astype(int),
        index=df.index,
        name="direction",
    )

    # 4. Strength 계산: direction * vol_scalar
    strength = pd.Series(
        position_shifted * vol_scalar_shifted,
        index=df.index,
        name="strength",
    )

    # 5. 숏 모드에 따른 시그널 처리
    if config.short_mode == ShortMode.DISABLED:
        # Long-Only: 모든 숏 시그널을 중립으로 변환
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    # HEDGE_ONLY는 페어 트레이딩에서 비일반적이므로 FULL과 동일 처리
    # (페어 트레이딩은 본질적으로 Long/Short 전략)

    # 6. 진입 시그널: 포지션이 0에서 non-zero로 변할 때
    prev_direction = direction.shift(1).fillna(0)

    long_entry_signal = (direction == Direction.LONG) & (prev_direction != Direction.LONG)
    short_entry_signal = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)

    entries = pd.Series(
        long_entry_signal | short_entry_signal,
        index=df.index,
        name="entries",
    )

    # 7. 청산 시그널: 포지션이 non-zero에서 0으로 변할 때, 또는 방향 반전
    to_neutral = (direction == Direction.NEUTRAL) & (prev_direction != Direction.NEUTRAL)
    reversal = direction * prev_direction < 0  # 부호가 바뀌면 반전

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
            "Copula Pairs Signal Stats | Total: %d, Long: %d (%.1f%%), Short: %d (%.1f%%)",
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
