"""Ensemble Signal Generator.

서브 전략들의 시그널을 수집하고 aggregator로 결합하여
앙상블 entries/exits/direction/strength를 생성합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from src.strategy.ensemble import aggregators
from src.strategy.ensemble.config import AggregationMethod, ShortMode
from src.strategy.types import Direction

if TYPE_CHECKING:
    from src.strategy.base import BaseStrategy
    from src.strategy.ensemble.config import EnsembleConfig
    from src.strategy.types import StrategySignals

logger = logging.getLogger(__name__)


def _collect_sub_signals(
    df: pd.DataFrame,
    sub_strategies: list[BaseStrategy],
    strategy_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """서브 전략들의 시그널을 수집.

    각 전략에 df.copy()를 전달하여 컬럼 충돌을 방지합니다.
    실패한 전략은 건너뛰고, 최소 1개 성공이 필요합니다.

    Args:
        df: OHLCV DataFrame
        sub_strategies: 서브 전략 인스턴스 목록
        strategy_names: 서브 전략 이름 목록

    Returns:
        (directions, strengths) DataFrame 튜플

    Raises:
        RuntimeError: 모든 서브 전략이 실패한 경우
    """
    dir_dict: dict[str, pd.Series] = {}
    str_dict: dict[str, pd.Series] = {}

    for name, strategy in zip(strategy_names, sub_strategies, strict=True):
        try:
            _, signals = strategy.run(df.copy())
            dir_dict[name] = signals.direction
            str_dict[name] = signals.strength
        except Exception:
            logger.warning("Sub-strategy '%s' failed, skipping", name, exc_info=True)

    if not dir_dict:
        msg = "All sub-strategies failed"
        raise RuntimeError(msg)

    directions = pd.DataFrame(dir_dict, index=df.index)
    strengths = pd.DataFrame(str_dict, index=df.index)

    logger.info(
        "Collected signals from %d/%d sub-strategies",
        len(dir_dict),
        len(sub_strategies),
    )

    return directions, strengths


def _apply_aggregation(
    directions: pd.DataFrame,
    strengths: pd.DataFrame,
    weights: pd.Series,
    config: EnsembleConfig,
) -> tuple[pd.Series, pd.Series]:
    """설정에 따라 적절한 aggregator 호출.

    Args:
        directions: (n_bars, n_strategies) direction 행렬
        strengths: (n_bars, n_strategies) strength 행렬
        weights: (n_strategies,) 정적 가중치
        config: 앙상블 설정

    Returns:
        (combined_direction, combined_strength) 튜플
    """
    if config.aggregation == AggregationMethod.INVERSE_VOLATILITY:
        return aggregators.inverse_volatility(
            directions, strengths, weights, vol_lookback=config.vol_lookback
        )
    if config.aggregation == AggregationMethod.MAJORITY_VOTE:
        return aggregators.majority_vote(
            directions, strengths, weights, min_agreement=config.min_agreement
        )
    if config.aggregation == AggregationMethod.STRATEGY_MOMENTUM:
        return aggregators.strategy_momentum(
            directions,
            strengths,
            weights,
            momentum_lookback=config.momentum_lookback,
            top_n=config.top_n,
        )
    # default: EQUAL_WEIGHT
    return aggregators.equal_weight(directions, strengths, weights)


def generate_signals(
    df: pd.DataFrame,
    sub_strategies: list[BaseStrategy],
    strategy_names: list[str],
    weights: pd.Series,
    config: EnsembleConfig,
) -> StrategySignals:
    """앙상블 시그널 생성.

    1. 각 서브 전략 run() → direction/strength 수집
    2. Aggregator 호출 → combined direction/strength
    3. vol_scalar 적용
    4. short_mode 필터링
    5. entries/exits 도출

    Args:
        df: 전처리된 DataFrame (vol_scalar 포함)
        sub_strategies: 서브 전략 인스턴스 목록
        strategy_names: 서브 전략 이름 목록
        weights: 정적 가중치 Series
        config: 앙상블 설정

    Returns:
        StrategySignals NamedTuple
    """
    from src.strategy.types import StrategySignals

    # 1. 서브 시그널 수집
    directions, strengths = _collect_sub_signals(df, sub_strategies, strategy_names)

    # 2. Aggregation
    combined_direction, combined_strength = _apply_aggregation(
        directions, strengths, weights, config
    )

    # 3. vol_scalar 적용
    vol_scalar: pd.Series = df["vol_scalar"]  # type: ignore[assignment]
    combined_strength = combined_strength * vol_scalar

    # NaN → 0 처리
    combined_direction = combined_direction.fillna(0).astype(int)
    combined_strength = combined_strength.fillna(0.0)

    # 4. Short mode 처리
    if config.short_mode == ShortMode.DISABLED:
        short_mask = combined_direction == Direction.SHORT
        combined_direction = combined_direction.where(~short_mask, Direction.NEUTRAL)
        combined_strength = combined_strength.where(~short_mask, 0.0)

    # 5. Entry/Exit 시그널 생성
    prev_direction = combined_direction.shift(1).fillna(0)

    long_entry = (combined_direction == Direction.LONG) & (prev_direction != Direction.LONG)
    short_entry = (combined_direction == Direction.SHORT) & (prev_direction != Direction.SHORT)
    entries = pd.Series(long_entry | short_entry, index=df.index, name="entries")

    to_neutral = (combined_direction == Direction.NEUTRAL) & (prev_direction != Direction.NEUTRAL)
    reversal = combined_direction * prev_direction < 0
    exits = pd.Series(to_neutral | reversal, index=df.index, name="exits")

    return StrategySignals(
        entries=entries,
        exits=exits,
        direction=combined_direction,
        strength=combined_strength,
    )
