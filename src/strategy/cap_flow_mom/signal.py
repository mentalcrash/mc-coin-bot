"""Capital Flow Momentum 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.

Signal Logic:
    1. fast_roc & slow_roc 동일 방향 → 모멘텀 확인
    2. Stablecoin ROC 방향이 모멘텀과 정렬 → 확신도 boost
    3. Stablecoin ROC 방향이 모멘텀과 괴리 → 확신도 dampen
    4. strength = direction * vol_scalar * conviction
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.cap_flow_mom.config import CapFlowMomConfig


def generate_signals(df: pd.DataFrame, config: CapFlowMomConfig) -> StrategySignals:
    """Capital Flow Momentum 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.cap_flow_mom.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    fast_roc = df["fast_roc"].shift(1)
    slow_roc = df["slow_roc"].shift(1)
    stab_roc = df["stablecoin_roc"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)
    dd: pd.Series = df["drawdown"].shift(1)  # type: ignore[assignment]

    # --- Dual-Speed ROC Momentum ---
    # 두 ROC가 threshold 이상 양수 → long signal
    # 두 ROC가 threshold 이하 음수 → short signal
    threshold = config.roc_threshold
    long_signal = (fast_roc > threshold) & (slow_roc > threshold)
    short_signal = (fast_roc < -threshold) & (slow_roc < -threshold)

    # --- Capital Flow Conviction ---
    # stablecoin_roc > 0: 자본 유입 (long 정렬 시 boost)
    # stablecoin_roc < 0: 자본 유출 (short 정렬 시 boost)
    # NaN → 중립(1.0)
    stab_positive = stab_roc > 0
    stab_negative = stab_roc < 0
    stab_available = stab_roc.notna()

    # 기본 conviction = 1.0
    conviction = pd.Series(1.0, index=df.index)

    # Long + 자본유입 정렬 → boost
    conviction = pd.Series(
        np.where(
            long_signal & stab_positive & stab_available,
            config.stablecoin_boost,
            np.where(
                long_signal & stab_negative & stab_available,
                config.stablecoin_dampen,
                np.where(
                    short_signal & stab_negative & stab_available,
                    config.stablecoin_boost,
                    np.where(
                        short_signal & stab_positive & stab_available,
                        config.stablecoin_dampen,
                        1.0,
                    ),
                ),
            ),
        ),
        index=df.index,
    )

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        drawdown_series=dd,
        config=config,
    )

    # --- Strength ---
    strength = direction.astype(float) * vol_scalar.fillna(0) * conviction

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
    drawdown_series: pd.Series,
    config: CapFlowMomConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.cap_flow_mom.config import ShortMode

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

    return pd.Series(raw, index=long_signal.index, dtype=int)
