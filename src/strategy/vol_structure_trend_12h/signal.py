"""Vol-Structure-Trend 12H signal generator.

3-scale 변동성 합의 x ROC 방향 -> 추세 시그널.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.vol_structure_trend_12h.config import ShortMode

_CONSENSUS_THRESHOLD = 0.33

if TYPE_CHECKING:
    from src.strategy.types import StrategySignals
    from src.strategy.vol_structure_trend_12h.config import VolStructureTrendConfig


def generate_signals(
    df: pd.DataFrame,
    config: VolStructureTrendConfig,
) -> StrategySignals:
    """시그널 생성 (shift(1) 적용).

    Args:
        df: preprocessed DataFrame.
        config: 전략 설정.

    Returns:
        StrategySignals (entries, exits, direction, strength).
    """
    from src.strategy.types import StrategySignals

    scales = (config.scale_short, config.scale_mid, config.scale_long)

    # --- Shift(1) 적용: 이전 bar 데이터 사용 ---
    vol_scalar = df["vol_scalar"].shift(1)
    roc_dir = df["roc_direction"].shift(1)

    # --- Multi-scale 합의 앙상블 ---
    signal_components: list[pd.Series] = []
    for s in scales:
        agreement = df[f"vol_agreement_{s}"].shift(1)
        # 합의 임계값 초과 시 방향 시그널 발생
        vol_confirmed = agreement >= config.vol_agreement_threshold
        sub_signal = pd.Series(
            np.where(
                vol_confirmed & (roc_dir > 0),
                1.0,
                np.where(vol_confirmed & (roc_dir < 0), -1.0, 0.0),
            ),
            index=df.index,
        )
        signal_components.append(sub_signal)

    # --- 3-scale 평균 합의 ---
    consensus = pd.concat(signal_components, axis=1).mean(axis=1)

    # --- Direction (ShortMode 분기) ---
    consensus_series: pd.Series = consensus  # type: ignore[assignment]
    direction = _compute_direction(consensus_series, df, config)

    # --- Strength ---
    strength = direction.astype(float) * consensus_series.abs() * vol_scalar.fillna(0)

    if config.short_mode == ShortMode.HEDGE_ONLY:
        strength = pd.Series(
            np.where(direction == -1, strength * config.hedge_strength_ratio, strength),
            index=df.index,
        )

    strength = strength.fillna(0.0)

    # --- Entries / Exits ---
    prev_dir = direction.shift(1).fillna(0).astype(int)
    entries = (direction != 0) & (direction != prev_dir)
    exits = (direction == 0) & (prev_dir != 0)

    return StrategySignals(
        entries=entries.astype(bool),
        exits=exits.astype(bool),
        direction=direction,
        strength=strength,
    )


def _compute_direction(
    consensus: pd.Series,
    df: pd.DataFrame,
    config: VolStructureTrendConfig,
) -> pd.Series:
    """ShortMode 분기 처리.

    Args:
        consensus: 3-scale 합의 시그널 (-1 ~ 1).
        df: preprocessed DataFrame.
        config: 전략 설정.

    Returns:
        Direction Series (-1, 0, 1).
    """
    long_signal = consensus > _CONSENSUS_THRESHOLD
    short_signal = consensus < -_CONSENSUS_THRESHOLD

    if config.short_mode == ShortMode.DISABLED:
        raw = np.where(long_signal, 1, 0)
    elif config.short_mode == ShortMode.HEDGE_ONLY:
        dd = df["drawdown"].shift(1)
        hedge_active = dd < config.hedge_threshold
        raw = np.where(
            long_signal,
            1,
            np.where(short_signal & hedge_active, -1, 0),
        )
    else:  # FULL
        raw = np.where(long_signal, 1, np.where(short_signal, -1, 0))

    return pd.Series(raw, index=df.index, dtype=int)
