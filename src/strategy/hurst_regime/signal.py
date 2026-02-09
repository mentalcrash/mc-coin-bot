"""Hurst/ER Regime Signal Generator.

전처리된 데이터에서 regime 판별 후 매매 시그널을 생성합니다.
Trending regime에서는 momentum following, mean-reverting regime에서는 z-score fading.

Signal Formula:
    1. Regime Classification: ER/Hurst thresholds 기반
    2. Direction: trending → sign(momentum), MR → -sign(z_score)
    3. Strength: direction * vol_scalar (변동성 스케일링 포함)

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

from src.strategy.hurst_regime.config import ShortMode
from src.strategy.types import Direction, StrategySignals

if TYPE_CHECKING:
    from src.strategy.hurst_regime.config import HurstRegimeConfig


def generate_signals(
    df: pd.DataFrame,
    config: HurstRegimeConfig | None = None,
) -> StrategySignals:
    """Hurst/ER Regime 시그널 생성.

    전처리된 DataFrame에서 regime을 판별하고 진입/청산 시그널을 생성합니다.
    Shift(1) Rule을 적용하여 미래 참조 편향을 방지합니다.

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: er, hurst, momentum, z_score, vol_scalar
        config: Hurst/ER Regime 설정. None이면 기본 설정 사용.

    Returns:
        StrategySignals NamedTuple:
            - entries: 진입 시그널 (bool Series)
            - exits: 청산 시그널 (bool Series)
            - direction: 방향 시리즈 (-1, 0, 1)
            - strength: 시그널 강도 (레버리지 무제한)

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    if config is None:
        from src.strategy.hurst_regime.config import HurstRegimeConfig

        config = HurstRegimeConfig()

    # 입력 검증
    required_cols = {"er", "hurst", "momentum", "z_score", "vol_scalar"}

    # HEDGE_ONLY 모드에서는 drawdown 컬럼 필요
    if config.short_mode == ShortMode.HEDGE_ONLY:
        required_cols.add("drawdown")

    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 1. Shift(1) 적용: 전봉 기준 시그널 (미래 참조 편향 방지)
    er_prev: pd.Series = df["er"].shift(1)  # type: ignore[assignment]
    hurst_prev: pd.Series = df["hurst"].shift(1)  # type: ignore[assignment]
    momentum_prev: pd.Series = df["momentum"].shift(1)  # type: ignore[assignment]
    z_score_prev: pd.Series = df["z_score"].shift(1)  # type: ignore[assignment]
    vol_scalar_prev: pd.Series = df["vol_scalar"].shift(1)  # type: ignore[assignment]

    # 2. Regime classification
    is_trending = (er_prev > config.er_trend_threshold) | (
        hurst_prev > config.hurst_trend_threshold
    )
    is_mr = (er_prev < config.er_mr_threshold) | (hurst_prev < config.hurst_mr_threshold)

    # 3. Direction per regime
    mom_dir = np.sign(momentum_prev)
    mr_dir = -np.sign(z_score_prev)

    # Trending → momentum following, MR → z-score fading, Neutral → reduced momentum
    direction_raw = np.where(is_trending, mom_dir, np.where(is_mr, mr_dir, mom_dir * 0.5))

    # 4. Strength (direction * vol_scalar)
    strength_raw = pd.Series(direction_raw * vol_scalar_prev, index=df.index)

    direction_sign = pd.Series(np.sign(strength_raw), index=df.index)
    direction = pd.Series(
        direction_sign.fillna(0).astype(int),
        index=df.index,
        name="direction",
    )
    strength = pd.Series(
        strength_raw.fillna(0),
        index=df.index,
        name="strength",
    )

    # 5. 숏 모드에 따른 시그널 처리
    if config.short_mode == ShortMode.DISABLED:
        # Long-Only: 모든 숏 시그널을 중립으로 변환
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        # 헤지 모드: 드로다운 임계값 초과 시에만 숏 허용
        drawdown_series: pd.Series = df["drawdown"]  # type: ignore[assignment]
        hedge_active = drawdown_series < config.hedge_threshold

        # 헤지 비활성 시 숏 -> 중립
        short_mask = direction == Direction.SHORT
        suppress_short = short_mask & ~hedge_active
        direction = direction.where(~suppress_short, Direction.NEUTRAL)
        strength = strength.where(~suppress_short, 0.0)

        # 헤지 활성 시 숏 강도 조절
        active_short = short_mask & hedge_active
        strength = strength.where(
            ~active_short,
            strength * config.hedge_strength_ratio,
        )

        # 헤지 활성화 통계 로깅
        hedge_days = int(hedge_active.sum())
        if hedge_days > 0:
            logger.info(
                "Hedge Mode | Active: {} days ({:.1f}%), Threshold: {:.1f}%",
                hedge_days,
                hedge_days / len(hedge_active) * 100,
                config.hedge_threshold * 100,
            )

    # else: ShortMode.FULL - 모든 시그널 그대로 유지

    # 6. 진입 시그널: 포지션이 변경될 때
    prev_direction = direction.shift(1).fillna(0)

    # Long 진입
    long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)
    # Short 진입
    short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)

    entries = pd.Series(
        long_entry | short_entry,
        index=df.index,
        name="entries",
    )

    # 7. 청산 시그널: 포지션이 중립으로 변할 때 또는 방향 반전
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
        trending_days = int(is_trending.sum())
        mr_days = int(is_mr.sum())
        logger.info(
            "Hurst/ER Regime | Trending: {} days, MR: {} days",
            trending_days,
            mr_days,
        )
        logger.info(
            "Hurst/ER Signals | Total: {}, Long: {} ({:.1f}%), Short: {} ({:.1f}%)",
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
