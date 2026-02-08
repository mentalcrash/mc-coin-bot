"""ADX Regime Filter Signal Generator.

ADX 기반으로 momentum과 mean-reversion 시그널을 블렌딩하여
레짐 적응형 시그널을 생성합니다.

Signal Formula:
    1. trend_weight = clip((adx - adx_low) / (adx_high - adx_low), 0, 1)
    2. mr_weight = 1 - trend_weight
    3. blended = trend_weight * mom_direction + mr_weight * mr_direction
    4. strength = blended * vol_scalar (Shift(1) 적용)

Regime Behavior:
    - ADX < adx_low: MR only (Z-Score 기반 역추세)
    - ADX > adx_high: Trend only (VW 모멘텀 추종)
    - adx_low <= ADX <= adx_high: 선형 블렌딩

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.tsmom.config import ShortMode
from src.strategy.types import Direction, StrategySignals

# Regime classification thresholds (near 0 or 1)
_TREND_ONLY_THRESHOLD = 0.99
_MR_ONLY_THRESHOLD = 0.01

if TYPE_CHECKING:
    from src.strategy.adx_regime.config import ADXRegimeConfig


def generate_signals(
    df: pd.DataFrame,
    config: ADXRegimeConfig | None = None,
) -> StrategySignals:
    """ADX Regime Filter 시그널 생성.

    전처리된 DataFrame에서 레짐 적응형 진입/청산 시그널과 강도를 계산합니다.
    Shift(1) Rule을 적용하여 미래 참조 편향을 방지합니다.

    Signal Generation Pipeline:
        1. ADX 기반 regime weight 계산 (Shift(1) 적용)
        2. Momentum direction 계산 (Shift(1) 적용)
        3. MR direction 계산 (Shift(1) 적용)
        4. 블렌딩 + vol_scalar 적용
        5. ShortMode 처리
        6. Entry/Exit 시그널 생성

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: adx, vw_momentum, z_score, vol_scalar
        config: ADX Regime 설정

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    if config is None:
        from src.strategy.adx_regime.config import ADXRegimeConfig

        config = ADXRegimeConfig()

    # 입력 검증
    required_cols = {"adx", "vw_momentum", "z_score", "vol_scalar"}
    if config.short_mode == ShortMode.HEDGE_ONLY:
        required_cols.add("drawdown")
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # ================================================================
    # 1. ADX Regime Weight (Shift(1) 적용)
    # ================================================================
    adx_series: pd.Series = df["adx"]  # type: ignore[assignment]
    adx_prev: pd.Series = adx_series.shift(1)  # type: ignore[assignment]

    # 선형 블렌딩: trend_weight = clip((adx - low) / (high - low), 0, 1)
    adx_range = config.adx_high - config.adx_low
    trend_weight = np.where(
        adx_prev >= config.adx_high,
        1.0,
        np.where(
            adx_prev < config.adx_low,
            0.0,
            (adx_prev - config.adx_low) / adx_range,
        ),
    )
    mr_weight = 1.0 - trend_weight

    # ================================================================
    # 2. Momentum Signal (Shift(1) 적용)
    # ================================================================
    vw_momentum: pd.Series = df["vw_momentum"]  # type: ignore[assignment]
    mom_direction = np.sign(vw_momentum.shift(1))

    # ================================================================
    # 3. Mean-Reversion Signal (Z-Score, Shift(1) 적용)
    # ================================================================
    z_score: pd.Series = df["z_score"]  # type: ignore[assignment]
    z_prev: pd.Series = z_score.shift(1)  # type: ignore[assignment]

    # 과매도(z < -entry) → 롱, 과매수(z > entry) → 숏
    mr_long = (z_prev < -config.mr_entry_z).astype(float)
    mr_short = (z_prev > config.mr_entry_z).astype(float) * -1
    mr_direction = mr_long + mr_short

    # ================================================================
    # 4. Blended Signal + Vol Scalar
    # ================================================================
    blended = trend_weight * mom_direction + mr_weight * mr_direction

    # Vol scalar (Shift(1) 적용)
    vol_scalar: pd.Series = df["vol_scalar"]  # type: ignore[assignment]
    strength_raw = pd.Series(
        blended * vol_scalar.shift(1),
        index=df.index,
    )

    # Direction & Strength
    direction_raw = pd.Series(np.sign(strength_raw), index=df.index)
    direction = pd.Series(
        direction_raw.fillna(0).astype(int),
        index=df.index,
        name="direction",
    )
    strength = pd.Series(
        strength_raw.fillna(0),
        index=df.index,
        name="strength",
    )

    # ================================================================
    # 5. ShortMode 처리
    # ================================================================
    if config.short_mode == ShortMode.DISABLED:
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        drawdown_series: pd.Series = df["drawdown"]  # type: ignore[assignment]
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
                "Hedge Mode | Active: %d days (%.1f%%), Threshold: %.1f%%",
                hedge_days,
                hedge_days / len(hedge_active) * 100,
                config.hedge_threshold * 100,
            )

    # else: ShortMode.FULL — 모든 시그널 그대로 유지

    # ================================================================
    # 6. Entry/Exit 시그널 생성
    # ================================================================
    prev_direction = direction.shift(1).fillna(0)

    long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)
    short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)
    entries = pd.Series(long_entry | short_entry, index=df.index, name="entries")

    to_neutral = (direction == Direction.NEUTRAL) & (prev_direction != Direction.NEUTRAL)
    reversal = direction * prev_direction < 0
    exits = pd.Series(to_neutral | reversal, index=df.index, name="exits")

    # 시그널 통계 로깅
    valid_strength = strength[strength != 0]
    long_signals = strength[strength > 0]
    short_signals = strength[strength < 0]

    if len(valid_strength) > 0:
        # regime 통계
        trend_days = int(np.sum(np.asarray(trend_weight) >= _TREND_ONLY_THRESHOLD))
        mr_days = int(np.sum(np.asarray(trend_weight) <= _MR_ONLY_THRESHOLD))
        blend_days = len(trend_weight) - trend_days - mr_days
        logger.info(
            "ADX Regime | Trend: %d days, MR: %d days, Blend: %d days",
            trend_days,
            mr_days,
            blend_days,
        )
        logger.info(
            "ADX-Regime Signals | Total: %d, Long: %d (%.1f%%), Short: %d (%.1f%%)",
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
