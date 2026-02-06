"""BB+RSI Mean Reversion Signal Generator.

볼린저밴드와 RSI를 조합한 평균회귀 시그널을 생성합니다.
ADX 필터로 추세장에서 시그널을 억제하고, 횡보장에서 활성화됩니다.

Signal Formula:
    1. bb_signal = -bb_position * 2 (밴드 하단 = 양수, 상단 = 음수)
    2. rsi_signal = (50 - RSI) / 50 (과매도 = 양수, 과매수 = 음수)
    3. combined = bb_weight * bb_signal + rsi_weight * rsi_signal
    4. strength = combined.shift(1) * vol_scalar.shift(1)

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.bb_rsi.config import BBRSIConfig, ShortMode
from src.strategy.types import Direction, StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: BBRSIConfig | None = None,
) -> StrategySignals:
    """BB+RSI 평균회귀 시그널 생성.

    전처리된 DataFrame에서 평균회귀 진입/청산 시그널과 강도를 계산합니다.

    Signal Generation Pipeline:
        1. BB position + RSI 조합으로 mean reversion signal 생성
        2. Shift(1) 적용 (미래 참조 편향 방지)
        3. Vol scalar 적용 (변동성 기반 포지션 사이징)
        4. ADX 필터 (추세장에서 포지션 축소)
        5. ShortMode 처리
        6. Entry/Exit 시그널 생성

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: bb_position, rsi, vol_scalar
        config: BB+RSI 설정

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    if config is None:
        config = BBRSIConfig()

    # 입력 검증
    required_cols = {"bb_position", "rsi", "vol_scalar"}
    if config.short_mode == ShortMode.HEDGE_ONLY:
        required_cols.add("drawdown")
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # ================================================================
    # 1. Mean Reversion Signal 계산
    # ================================================================
    bb_position: pd.Series = df["bb_position"]  # type: ignore[assignment]
    rsi_series: pd.Series = df["rsi"]  # type: ignore[assignment]
    vol_scalar: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    # BB signal: 밴드 하단이면 양수(롱), 상단이면 음수(숏)
    # bb_position은 (close - middle) / bandwidth → 평균회귀이므로 역수
    bb_signal: pd.Series = -bb_position * 2  # type: ignore[assignment]

    # RSI signal: 과매도면 양수(롱), 과매수면 음수(숏)
    rsi_signal: pd.Series = (50 - rsi_series) / 50  # type: ignore[assignment]

    # 가중 합산
    combined: pd.Series = (  # type: ignore[assignment]
        config.bb_weight * bb_signal + config.rsi_weight * rsi_signal
    )

    # ================================================================
    # 2. Shift(1) + Vol Scalar 적용
    # ================================================================
    combined_shifted: pd.Series = combined.shift(1)  # type: ignore[assignment]
    vol_scalar_shifted: pd.Series = vol_scalar.shift(1)  # type: ignore[assignment]

    # 최종 strength = combined signal * vol_scalar
    raw_strength: pd.Series = combined_shifted * vol_scalar_shifted  # type: ignore[assignment]

    # ================================================================
    # 3. Direction & Strength
    # ================================================================
    direction_raw = pd.Series(np.sign(raw_strength), index=df.index)
    direction = pd.Series(
        direction_raw.fillna(0).astype(int),
        index=df.index,
        name="direction",
    )
    strength = pd.Series(
        raw_strength.fillna(0),
        index=df.index,
        name="strength",
    )

    # ================================================================
    # 4. ADX 레짐 필터 (추세장에서 포지션 축소 — TSMOM과 반대 방향!)
    # ================================================================
    if config.use_adx_filter and "adx" in df.columns:
        adx_series: pd.Series = df["adx"].shift(1)  # type: ignore[assignment]
        # ADX >= threshold = 추세장 → 평균회귀에 불리 → 포지션 축소
        trending_mask = adx_series >= config.adx_threshold
        strength = strength.where(
            ~trending_mask,
            strength * config.trending_position_scale,
        )

        trending_days = int(trending_mask.sum())
        if trending_days > 0:
            logger.info(
                "📊 ADX Filter | Trending: %d days (%.1f%%), ADX >= %.0f, Scale: %.0f%%",
                trending_days,
                trending_days / len(trending_mask) * 100,
                config.adx_threshold,
                config.trending_position_scale * 100,
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
                "🛡️ Hedge Mode | Active: %d days (%.1f%%), Threshold: %.1f%%",
                hedge_days,
                hedge_days / len(hedge_active) * 100,
                config.hedge_threshold * 100,
            )

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
        logger.info(
            "📊 BB-RSI Signals | Total: %d, Long: %d (%.1f%%), Short: %d (%.1f%%)",
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
