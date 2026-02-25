"""Disposition CGO 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.

Signal Logic (Grinblatt-Han 2005 + Frazzini 2006):
    1. CGO(smoothed) > cgo_entry_threshold → 미실현 이익 → disposition underreaction → Long
    2. CGO(smoothed) < -cgo_entry_threshold → 미실현 손실 → loss aversion → Short
    3. Overhang spread MA > spread_confirm_threshold → momentum-disposition alignment 확인
    4. 두 시그널 교차 확인: CGO 방향 + spread 방향 일치 시에만 진입
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.disposition_cgo.config import DispositionCgoConfig


def generate_signals(df: pd.DataFrame, config: DispositionCgoConfig) -> StrategySignals:
    """Disposition CGO 시그널 생성.

    Signal Logic:
        - cgo_smooth > threshold AND spread_ma > spread_threshold → Long
          (미실현 이익 + momentum alignment = underreaction drift)
        - cgo_smooth < -threshold AND spread_ma > spread_threshold → Short
          (미실현 손실 + momentum alignment = underreaction drift, 반대 방향)
        - 조건 미충족 → Neutral

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.disposition_cgo.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    cgo_smooth = df["cgo_smooth"].shift(1)
    spread_ma = df["overhang_spread_ma"].shift(1)
    momentum = df["momentum"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    # CGO 방향: 양의 CGO = 미실현 이익 (매도 저항 = momentum 지속)
    # 음의 CGO = 미실현 손실 (loss aversion = 역방향 momentum)
    long_cgo = cgo_smooth > config.cgo_entry_threshold
    short_cgo = cgo_smooth < -config.cgo_entry_threshold

    # Spread confirmation: CGO-momentum alignment 확인
    # spread_ma > threshold → CGO와 momentum 방향이 일치
    spread_confirm = spread_ma > config.spread_confirm_threshold

    # 2시그널 교차 확인: CGO 방향 + momentum alignment
    # 추가: momentum 방향으로 최종 필터링
    long_signal = long_cgo & spread_confirm & (momentum > 0)
    short_signal = short_cgo & spread_confirm & (momentum < 0)

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength ---
    strength = direction.astype(float) * vol_scalar.fillna(0)

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
    long_signal: pd.Series,
    short_signal: pd.Series,
    df: pd.DataFrame,
    config: DispositionCgoConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.disposition_cgo.config import ShortMode

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
