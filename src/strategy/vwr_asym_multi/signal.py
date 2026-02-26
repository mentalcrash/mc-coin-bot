"""VWR Asymmetric Multi-Scale 시그널 생성.

3-scale VWR z-score consensus + 비대칭 long/short 임계값으로 시그널을 생성한다.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.vwr_asym_multi.config import VwrAsymMultiConfig


def generate_signals(
    df: pd.DataFrame,
    config: VwrAsymMultiConfig,
) -> StrategySignals:
    """VWR Asymmetric Multi-Scale 시그널 생성.

    Signal Logic:
        1. 각 lookback(10/21/42)의 VWR z-score를 shift(1)
        2. 3-scale z-score 평균(consensus) 산출
        3. consensus > long_threshold → direction = +1
           consensus < -short_threshold → direction = -1 (비대칭)
        4. strength = |consensus| * vol_scalar

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.vwr_asym_multi.config import ShortMode

    lookbacks = (config.lookback_short, config.lookback_mid, config.lookback_long)

    # --- Shift(1): 전봉 기준 시그널 ---
    vol_scalar = df["vol_scalar"].shift(1)
    dd = df["drawdown"].shift(1)

    # --- Per-scale VWR z-score (shifted) ---
    zscore_components: list[pd.Series] = []
    for lb in lookbacks:
        zscore_shifted: pd.Series = df[f"vwr_zscore_{lb}"].shift(1)  # type: ignore[assignment]
        zscore_components.append(zscore_shifted)

    # --- Consensus: 3-scale z-score 평균 ---
    consensus: pd.Series = pd.concat(zscore_components, axis=1).mean(axis=1)  # type: ignore[assignment]

    # --- Direction (비대칭 임계값 + ShortMode 분기) ---
    dd_series: pd.Series = dd  # type: ignore[assignment]
    direction = _compute_direction(consensus, dd_series, config)

    # --- Strength ---
    abs_consensus: pd.Series = consensus.abs()  # type: ignore[assignment]
    strength = direction.astype(float) * abs_consensus * vol_scalar.fillna(0)

    if config.short_mode == ShortMode.HEDGE_ONLY:
        strength = pd.Series(
            np.where(
                direction == -1,
                strength * config.hedge_strength_ratio,
                strength,
            ),
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
    drawdown_series: pd.Series,
    config: VwrAsymMultiConfig,
) -> pd.Series:
    """비대칭 임계값 + ShortMode 3-way 분기로 direction 계산.

    비대칭 핵심: long_threshold < short_threshold로
    crypto의 구조적 long drift (positive skew)를 반영.
    숏 진입은 더 강한 확신(높은 임계값)이 필요하다.
    """
    from src.strategy.vwr_asym_multi.config import ShortMode

    # 비대칭 임계값 적용
    long_signal = consensus > config.long_threshold
    short_signal = consensus < -config.short_threshold

    if config.short_mode == ShortMode.DISABLED:
        raw = np.where(long_signal, 1, 0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        hedge_active = drawdown_series < config.hedge_threshold
        raw = np.where(
            long_signal,
            1,
            np.where(short_signal & hedge_active, -1, 0),
        )

    else:  # FULL
        raw = np.where(long_signal, 1, np.where(short_signal, -1, 0))

    return pd.Series(raw, index=consensus.index, dtype=int)
