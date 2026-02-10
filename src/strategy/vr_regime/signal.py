"""Variance Ratio Regime Signal Generator.

VR test로 regime 분류 후 momentum/contrarian 전환.

Signal Formula:
    1. Shift(1) on vr, vr_z_stat, mom_direction, vol_scalar
    2. Trending: VR > 1 AND z > significance_z → follow momentum
    3. Mean-reverting: VR < 1 AND z < -significance_z → contrarian
    4. Random walk: else → neutral
    5. strength = direction * vol_scalar

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.types import Direction, StrategySignals
from src.strategy.vr_regime.config import ShortMode, VRRegimeConfig


def generate_signals(
    df: pd.DataFrame,
    config: VRRegimeConfig | None = None,
) -> StrategySignals:
    """Variance Ratio Regime 시그널 생성.

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
        config: VR Regime 설정. None이면 기본 설정 사용.

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    if config is None:
        config = VRRegimeConfig()

    required_cols = {"vr", "vr_z_stat", "mom_direction", "vol_scalar"}
    if config.short_mode == ShortMode.HEDGE_ONLY:
        required_cols.add("drawdown")

    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 1. Shift(1) 적용: 전봉 기준 시그널
    vr_prev: pd.Series = df["vr"].shift(1)  # type: ignore[assignment]
    z_prev: pd.Series = df["vr_z_stat"].shift(1)  # type: ignore[assignment]
    mom_dir_prev: pd.Series = df["mom_direction"].shift(1)  # type: ignore[assignment]
    vol_scalar_prev: pd.Series = df["vol_scalar"].shift(1)  # type: ignore[assignment]

    # 2. Regime classification
    trending = (vr_prev > 1.0) & (z_prev > config.significance_z)
    mean_reverting = (vr_prev < 1.0) & (z_prev < -config.significance_z)

    # 3. Direction:
    #    Trending → follow momentum
    #    Mean-reverting → contrarian
    #    Random walk → neutral
    direction_raw = pd.Series(
        np.where(
            trending,
            mom_dir_prev,
            np.where(mean_reverting, -mom_dir_prev, 0),
        ),
        index=df.index,
    )

    # 4. Strength = direction * vol_scalar
    strength_raw = direction_raw * vol_scalar_prev

    # 5. Direction 정규화
    direction = pd.Series(
        np.sign(strength_raw).fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    # 6. 강도 계산
    strength = pd.Series(
        strength_raw.fillna(0),
        index=df.index,
        name="strength",
    )

    # 7. 숏 모드에 따른 시그널 처리
    if config.short_mode == ShortMode.DISABLED:
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        drawdown_series: pd.Series = df["drawdown"].shift(1)  # type: ignore[assignment]
        hedge_active = drawdown_series < config.hedge_threshold

        short_mask = direction == Direction.SHORT
        suppress_short = short_mask & ~hedge_active
        direction = direction.where(~suppress_short, Direction.NEUTRAL)
        strength = strength.where(~suppress_short, 0.0)

        active_short = short_mask & hedge_active
        strength = strength.where(
            ~active_short,
            strength * config.hedge_strength_ratio,
        )

        hedge_days = int(hedge_active.sum())
        if hedge_days > 0:
            logger.info(
                "Hedge Mode | Active: {} days ({:.1f}%), Threshold: {:.1f}%",
                hedge_days,
                hedge_days / len(hedge_active) * 100,
                config.hedge_threshold * 100,
            )

    # 8. 진입/청산 시그널
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

    # 디버그: 시그널 통계
    valid_strength = strength[strength != 0]
    long_signals = strength[strength > 0]
    short_signals = strength[strength < 0]

    if len(valid_strength) > 0:
        logger.info(
            "Signal Statistics | Total: {} signals, Long: {} ({:.1f}%), Short: {} ({:.1f}%)",
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
