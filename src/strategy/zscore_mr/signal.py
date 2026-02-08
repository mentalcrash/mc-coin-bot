"""Z-Score Mean Reversion Signal Generator.

동적 lookback z-score 기반 평균회귀 시그널을 생성합니다.
z-score가 극단적일 때 반대 방향으로 진입하고, 평균으로 복귀하면 청산합니다.

Signal Formula:
    1. z_score = df["zscore"].shift(1) (lookahead prevention)
    2. long_entry: z_score < -entry_z
    3. short_entry: z_score > entry_z
    4. exit: abs(z_score) < exit_z
    5. ffill for holding between entry and exit

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.types import Direction, StrategySignals
from src.strategy.zscore_mr.config import ShortMode, ZScoreMRConfig


def generate_signals(
    df: pd.DataFrame,
    config: ZScoreMRConfig | None = None,
) -> StrategySignals:
    """Z-Score 평균회귀 시그널 생성.

    전처리된 DataFrame에서 z-score 기반 평균회귀 진입/청산 시그널을 계산합니다.

    Signal Generation Pipeline:
        1. Z-score shift(1) (미래 참조 편향 방지)
        2. Entry/Exit threshold 판정
        3. ffill로 포지션 유지 (entry~exit 사이)
        4. Vol scalar 적용 (변동성 기반 포지션 사이징)
        5. ShortMode 처리
        6. Entry/Exit 시그널 생성

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: zscore, vol_scalar
        config: Z-Score MR 설정

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    if config is None:
        config = ZScoreMRConfig()

    # 입력 검증
    required_cols = {"zscore", "vol_scalar"}
    if config.short_mode == ShortMode.HEDGE_ONLY:
        required_cols.add("drawdown")
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # ================================================================
    # 1. Z-Score Shift(1) - 미래 참조 편향 방지
    # ================================================================
    z_score: pd.Series = df["zscore"].shift(1)  # type: ignore[assignment]
    vol_scalar: pd.Series = df["vol_scalar"].shift(1)  # type: ignore[assignment]

    # ================================================================
    # 2. Direction 판정 (entry/exit threshold)
    # ================================================================
    # z < -entry_z -> long (1), z > entry_z -> short (-1), |z| < exit_z -> neutral (0)
    # 그 외 (exit_z <= |z| <= entry_z) -> NaN (이전 포지션 유지)
    raw_direction = pd.Series(
        np.where(
            z_score < -config.entry_z,
            1.0,
            np.where(
                z_score > config.entry_z,
                -1.0,
                np.where(
                    z_score.abs() < config.exit_z,
                    0.0,
                    np.nan,
                ),
            ),
        ),
        index=df.index,
    )

    # ffill: entry~exit 사이에서 포지션 유지
    direction_filled = raw_direction.ffill().fillna(0)

    direction = pd.Series(
        direction_filled.astype(int),
        index=df.index,
        name="direction",
    )

    # ================================================================
    # 3. Strength = direction * vol_scalar
    # ================================================================
    strength = pd.Series(
        direction.astype(float) * vol_scalar.fillna(0),
        index=df.index,
        name="strength",
    )

    # ================================================================
    # 4. ShortMode 처리
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

    # ================================================================
    # 5. Entry/Exit 시그널 생성
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
        logger.info(
            "Z-Score MR Signals | Total: %d, Long: %d (%.1f%%), Short: %d (%.1f%%)",
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
