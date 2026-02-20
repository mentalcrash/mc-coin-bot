"""EMA Ribbon Momentum 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.ema_ribbon_mom.config import EmaRibbonMomConfig


def generate_signals(df: pd.DataFrame, config: EmaRibbonMomConfig) -> StrategySignals:
    """EMA 리본 정렬도 + ROC 모멘텀으로 시그널을 생성한다.

    진입 조건:
    - |alignment| >= threshold (리본이 충분히 정렬)
    - ROC 방향이 alignment 방향과 일치
    - ribbon_direction이 alignment과 같은 방향

    Args:
        df: preprocess()로 지표가 추가된 DataFrame.
        config: 전략 설정.

    Returns:
        StrategySignals.
    """
    from src.strategy.ema_ribbon_mom.config import ShortMode

    # --- Shift(1): 전봉 기준 ---
    alignment = df["alignment"].shift(1)
    roc_val = df["roc"].shift(1)
    ribbon_dir = df["ribbon_direction"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # Long: alignment >= threshold AND roc > 0 AND ribbon_direction > 0
    long_signal = (alignment >= config.alignment_threshold) & (roc_val > 0) & (ribbon_dir > 0)

    # Short: alignment <= -threshold AND roc < 0 AND ribbon_direction < 0
    short_signal = (alignment <= -config.alignment_threshold) & (roc_val < 0) & (ribbon_dir < 0)

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength: alignment 강도를 반영 ---
    alignment_weight = alignment.abs().clip(upper=1.0)
    strength = direction.astype(float) * vol_scalar.fillna(0) * alignment_weight.fillna(0)

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
    config: EmaRibbonMomConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.ema_ribbon_mom.config import ShortMode

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
