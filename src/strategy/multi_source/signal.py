"""Multi-Source 전략 시그널 생성.

전처리된 서브시그널들을 결합하여 최종 direction + strength를 생성합니다.
Shift(1) 적용으로 look-ahead bias를 방지합니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.strategy.multi_source.config import MultiSourceConfig
    from src.strategy.types import StrategySignals


def generate_signals(df: pd.DataFrame, config: MultiSourceConfig) -> StrategySignals:
    """시그널 생성 — 서브시그널 결합 + shift(1) + entries/exits.

    Args:
        df: 전처리된 DataFrame (_sub_0, _sub_1, ... 컬럼 포함)
        config: MultiSourceConfig 인스턴스

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.multi_source.config import SignalCombineMethod
    from src.strategy.types import StrategySignals

    # Shift(1): 전봉 기준으로 시그널 생성 (look-ahead bias 방지)
    sub_cols = [f"_sub_{i}" for i in range(len(config.signals))]
    shifted_subs: dict[str, pd.Series] = {  # type: ignore[type-arg]
        col: df[col].shift(1) for col in sub_cols
    }
    vol_scalar: pd.Series = df["_vol_scalar"].shift(1)  # type: ignore[assignment]

    # 결합 방법별 복합 시그널 계산
    if config.combine_method == SignalCombineMethod.ZSCORE_SUM:
        combined = _zscore_sum(shifted_subs, config)
    elif config.combine_method == SignalCombineMethod.MAJORITY_VOTE:
        combined = _majority_vote(shifted_subs, config)
    elif config.combine_method == SignalCombineMethod.WEIGHTED_SUM:
        combined = _weighted_sum(shifted_subs, config)
    else:
        msg = f"Unknown combine method: {config.combine_method}"
        raise ValueError(msg)

    # Direction 결정
    direction = _compute_direction(combined, config)

    # Strength: |combined| * vol_scalar
    base_strength = combined.abs() * vol_scalar.fillna(0)
    strength = direction.astype(float) * base_strength
    strength = strength.fillna(0.0)

    # Entries / Exits
    prev_dir = direction.shift(1).fillna(0).astype(int)
    entries = (direction != 0) & (direction != prev_dir)
    exits = (direction == 0) & (prev_dir != 0)

    return StrategySignals(
        entries=entries.astype(bool),
        exits=exits.astype(bool),
        direction=direction,
        strength=strength,
    )


def _zscore_sum(
    shifted_subs: Mapping[str, pd.Series],  # type: ignore[type-arg]
    config: MultiSourceConfig,
) -> pd.Series:  # type: ignore[type-arg]
    """Z-score 가중합."""
    weights = [spec.weight for spec in config.signals]
    total_weight = sum(weights)

    result = pd.Series(0.0, index=next(iter(shifted_subs.values())).index)
    for (_, sub), w in zip(shifted_subs.items(), weights, strict=False):
        result = result + sub.fillna(0) * (w / total_weight)

    return result


def _majority_vote(
    shifted_subs: Mapping[str, pd.Series],  # type: ignore[type-arg]
    config: MultiSourceConfig,
) -> pd.Series:  # type: ignore[type-arg]
    """다수결 투표 — 과반 합의 시 방향 결정."""
    n_signals = len(config.signals)
    index = next(iter(shifted_subs.values())).index

    votes_long = pd.Series(0, index=index)
    votes_short = pd.Series(0, index=index)

    for sub in shifted_subs.values():
        votes_long = votes_long + (sub > 0).astype(int)
        votes_short = votes_short + (sub < 0).astype(int)

    agreement_long = votes_long / n_signals
    agreement_short = votes_short / n_signals

    result = pd.Series(
        np.where(
            agreement_long >= config.min_agreement,
            agreement_long,
            np.where(agreement_short >= config.min_agreement, -agreement_short, 0.0),
        ),
        index=index,
    )

    return result


def _weighted_sum(
    shifted_subs: Mapping[str, pd.Series],  # type: ignore[type-arg]
    config: MultiSourceConfig,
) -> pd.Series:  # type: ignore[type-arg]
    """가중합 (정규화 없음)."""
    result = pd.Series(0.0, index=next(iter(shifted_subs.values())).index)
    for (_, sub), spec in zip(shifted_subs.items(), config.signals, strict=False):
        result = result + sub.fillna(0) * spec.weight

    return result


def _compute_direction(
    combined: pd.Series,  # type: ignore[type-arg]
    config: MultiSourceConfig,
) -> pd.Series:  # type: ignore[type-arg]
    """결합 시그널에서 direction 결정 (entry/exit threshold + ShortMode 적용)."""
    from src.strategy.multi_source.config import ShortMode

    abs_signal = combined.abs()
    long_signal = (combined > 0) & (abs_signal >= config.entry_threshold)
    short_signal = (combined < 0) & (abs_signal >= config.entry_threshold)
    in_position = abs_signal >= config.exit_threshold

    if config.short_mode == ShortMode.DISABLED:
        raw = np.where(long_signal & in_position, 1, 0)
    elif config.short_mode == ShortMode.HEDGE_ONLY:
        raw = np.where(
            long_signal & in_position,
            1,
            np.where(short_signal & in_position, -1, 0),
        )
    else:  # FULL
        raw = np.where(
            long_signal & in_position,
            1,
            np.where(short_signal & in_position, -1, 0),
        )

    return pd.Series(raw, index=combined.index, dtype=int)
