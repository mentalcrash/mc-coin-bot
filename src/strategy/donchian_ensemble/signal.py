"""Donchian Ensemble Signal Generator.

9개 lookback의 Donchian Channel breakout 시그널을 평균내어
앙상블 방향을 결정합니다.

Signal Formula:
    1. 각 lookback에 대해:
       signal_i = +1 if close > prev_dc_upper, -1 if close < prev_dc_lower, 0 otherwise
    2. ensemble = mean(all signal_i)
    3. direction = sign(ensemble)
    4. strength = ensemble * vol_scalar

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops on DataFrame rows)
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.donchian_ensemble.config import ShortMode
from src.strategy.types import Direction

if TYPE_CHECKING:
    from src.strategy.donchian_ensemble.config import DonchianEnsembleConfig
    from src.strategy.types import StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: DonchianEnsembleConfig | None = None,
) -> StrategySignals:
    """Donchian Ensemble 시그널 생성.

    각 lookback의 Donchian Channel에 대해 breakout 시그널(+1/0/-1)을
    계산하고, 전체 평균으로 앙상블 방향을 결정합니다.

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: close, dc_upper_{lb}, dc_lower_{lb}, vol_scalar
        config: 전략 설정. None이면 기본 설정 사용.

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    from src.strategy.donchian_ensemble.config import DonchianEnsembleConfig
    from src.strategy.types import StrategySignals

    if config is None:
        config = DonchianEnsembleConfig()

    # 입력 검증
    required_cols = {"close", "vol_scalar"}
    for lb in config.lookbacks:
        required_cols.add(f"dc_upper_{lb}")
        required_cols.add(f"dc_lower_{lb}")

    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 컬럼 추출
    close: pd.Series = df["close"]  # type: ignore[assignment]
    vol_scalar: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    # 1. 각 lookback에 대한 breakout 시그널 계산
    signal_components: list[pd.Series] = []
    for lb in config.lookbacks:
        dc_upper: pd.Series = df[f"dc_upper_{lb}"]  # type: ignore[assignment]
        dc_lower: pd.Series = df[f"dc_lower_{lb}"]  # type: ignore[assignment]

        # Shift(1): 전봉 채널 기준 (미래 참조 방지)
        prev_upper = dc_upper.shift(1)
        prev_lower = dc_lower.shift(1)

        # Breakout signal: +1 if close > prev_upper, -1 if close < prev_lower, 0 otherwise
        signal_i = pd.Series(
            np.where(
                close > prev_upper,
                1.0,
                np.where(close < prev_lower, -1.0, 0.0),
            ),
            index=df.index,
        )
        signal_components.append(signal_i)

    # 2. Ensemble: 전체 시그널의 평균
    ensemble = pd.concat(signal_components, axis=1).mean(axis=1)

    # 3. Direction 계산
    direction_raw = pd.Series(np.sign(ensemble), index=df.index)
    direction = pd.Series(
        direction_raw.fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    # 4. Strength 계산: ensemble * vol_scalar
    strength = pd.Series(
        (ensemble * vol_scalar).fillna(0.0),
        index=df.index,
        name="strength",
    )

    # 5. 숏 모드 처리
    if config.short_mode == ShortMode.DISABLED:
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

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
    logger = logging.getLogger(__name__)
    valid_strength = strength[strength != 0]
    long_signals = strength[strength > 0]
    short_signals = strength[strength < 0]

    if len(valid_strength) > 0:
        logger.info(
            "Donchian Ensemble Signals | Total: %d, Long: %d (%.1f%%), Short: %d (%.1f%%)",
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
