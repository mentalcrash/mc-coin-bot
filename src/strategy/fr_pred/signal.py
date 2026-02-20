"""FR-Pred 시그널 생성.

FR z-score 평균회귀 + FR momentum 이중 시그널.
Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.fr_pred.config import FRPredConfig


def generate_signals(df: pd.DataFrame, config: FRPredConfig) -> StrategySignals:
    """FR-Pred 시그널 생성.

    이중 시그널:
    1. Mean-reversion: FR zscore > threshold → 숏 (과열), < -threshold → 롱 (과매도)
    2. Momentum: FR fast MA > slow MA → 숏 캐리 (양+FR 확대), 반대 → 롱 캐리

    Args:
        df: preprocess() 출력 DataFrame.
        config: 전략 설정.

    Returns:
        StrategySignals.
    """
    from src.strategy.fr_pred.config import ShortMode

    # Shift(1): 전봉 기준
    fr_zscore = df["fr_zscore"].shift(1)
    fr_mom_cross = df["fr_mom_cross"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Mean-Reversion Signal ---
    # 양(+) FR z-score → 시장 과열 → 숏 (contrarian)
    # 음(-) FR z-score → 시장 과매도 → 롱 (contrarian)
    mr_long = fr_zscore < -config.fr_mr_threshold
    mr_short = fr_zscore > config.fr_mr_threshold
    mr_signal = pd.Series(
        np.where(mr_long, 1.0, np.where(mr_short, -1.0, 0.0)),
        index=df.index,
    )

    # --- Momentum Signal ---
    # FR fast < slow (FR 하락 추세) → 시장 냉각 → 롱 기회
    # FR fast > slow (FR 상승 추세) → 과열 확대 → 숏 기회
    mom_long = fr_mom_cross < 0
    mom_short = fr_mom_cross > 0
    mom_signal = pd.Series(
        np.where(mom_long, 1.0, np.where(mom_short, -1.0, 0.0)),
        index=df.index,
    )

    # --- Combined Signal ---
    combined = config.mr_weight * mr_signal + config.mom_weight * mom_signal

    # Direction
    raw_direction = np.sign(combined)
    long_signal = raw_direction > 0
    short_signal = raw_direction < 0

    direction = _compute_direction(long_signal, short_signal, df, config)

    # Strength: direction * vol_scalar * |combined|
    conviction = combined.abs().clip(upper=1.0).fillna(0.0)
    strength = direction.astype(float) * vol_scalar.fillna(0.0) * conviction

    if config.short_mode == ShortMode.HEDGE_ONLY:
        strength = pd.Series(
            np.where(direction == -1, strength * config.hedge_strength_ratio, strength),
            index=df.index,
        )

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


def _compute_direction(
    long_signal: pd.Series,
    short_signal: pd.Series,
    df: pd.DataFrame,
    config: FRPredConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.fr_pred.config import ShortMode

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
