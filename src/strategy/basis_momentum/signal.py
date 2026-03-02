"""Basis-Momentum 시그널 생성.

FR 변화율 z-score 기반 단일 시그널 (channel 전략과 달리 consensus 아님).

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.basis_momentum.config import BasisMomentumConfig


def generate_signals(df: pd.DataFrame, config: BasisMomentumConfig) -> StrategySignals:
    """Basis-Momentum 시그널 생성.

    Signal Logic:
        1. basis_mom > entry_zscore → LONG (FR 상향 가속 = 강세 펀딩 압력)
        2. basis_mom < -entry_zscore → SHORT (FR 하향 가속 = 약세 펀딩 압력)
        3. |basis_mom| < exit_zscore → FLAT (중립 복귀)
        4. Hysteresis: 진입 후 exit_zscore까지 유지 (entry_zscore 재돌파 불필요)

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.basis_momentum.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    basis_mom: pd.Series = df["basis_mom"].shift(1)  # type: ignore[assignment]
    vol_scalar: pd.Series = df["vol_scalar"].shift(1)  # type: ignore[assignment]
    dd: pd.Series = df["drawdown"].shift(1)  # type: ignore[assignment]

    # --- Threshold signals ---
    long_signal = basis_mom > config.entry_zscore
    short_signal = basis_mom < -config.entry_zscore
    exit_zone = basis_mom.abs() < config.exit_zscore

    # --- Direction (ShortMode 분기 + hysteresis) ---
    direction = _compute_direction_with_hysteresis(long_signal, short_signal, exit_zone, dd, config)

    # --- Strength: |z-score| / entry_zscore * vol_scalar ---
    abs_zscore: pd.Series = basis_mom.abs()  # type: ignore[assignment]
    strength = direction.astype(float) * (abs_zscore / config.entry_zscore) * vol_scalar.fillna(0)

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


def _compute_direction_with_hysteresis(
    long_signal: pd.Series,
    short_signal: pd.Series,
    exit_zone: pd.Series,
    drawdown_series: pd.Series,
    config: BasisMomentumConfig,
) -> pd.Series:
    """ShortMode 3-way 분기 + hysteresis로 direction 계산.

    Hysteresis 로직:
        - entry_zscore 돌파 → 진입
        - exit_zscore 이하 → 청산 (중립 복귀)
        - exit_zscore ~ entry_zscore 구간 → 기존 포지션 유지

    벡터화 불가 (상태 의존) → numpy 배열 순회로 구현.
    """
    from src.strategy.basis_momentum.config import ShortMode

    n = len(long_signal)
    direction = np.zeros(n, dtype=int)

    long_arr = long_signal.to_numpy()
    short_arr = short_signal.to_numpy()
    exit_arr = exit_zone.to_numpy()
    dd_arr = drawdown_series.to_numpy()

    prev = 0
    for i in range(n):
        if exit_arr[i]:
            # exit zone → 중립 복귀
            prev = 0
        elif long_arr[i]:
            prev = 1
        elif short_arr[i]:
            # ShortMode 분기
            if config.short_mode == ShortMode.DISABLED:
                prev = 0  # 숏 불가 → 중립
            elif config.short_mode == ShortMode.HEDGE_ONLY:
                dd_val = dd_arr[i]
                prev = -1 if not np.isnan(dd_val) and dd_val < config.hedge_threshold else 0
            else:  # FULL
                prev = -1
        # else: exit_zscore ~ entry_zscore 구간 → prev 유지

        direction[i] = prev

    return pd.Series(direction, index=long_signal.index, dtype=int)
