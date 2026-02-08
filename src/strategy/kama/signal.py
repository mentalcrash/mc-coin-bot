"""KAMA Trend Following Signal Generator.

KAMA와 ATR 필터를 조합한 추세 추종 시그널을 생성합니다.
KAMA가 상승하고 가격이 KAMA + ATR*mult 위에 있으면 롱,
KAMA가 하락하고 가격이 KAMA - ATR*mult 아래에 있으면 숏입니다.

Signal Formula:
    1. kama_rising = kama.shift(1) > kama.shift(2)
    2. long = (close.shift(1) > kama.shift(1) + atr_mult * atr.shift(1)) & kama_rising
    3. short = (close.shift(1) < kama.shift(1) - atr_mult * atr.shift(1)) & kama_falling
    4. strength = direction * vol_scalar.shift(1)

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.kama.config import KAMAConfig, ShortMode
from src.strategy.types import Direction, StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: KAMAConfig | None = None,
) -> StrategySignals:
    """KAMA 추세 추종 시그널 생성.

    전처리된 DataFrame에서 추세 추종 진입/청산 시그널과 강도를 계산합니다.

    Signal Generation Pipeline:
        1. KAMA 방향 판단 (rising/falling)
        2. ATR 필터 적용 (close vs KAMA +/- ATR*mult)
        3. Shift(1) 적용 (미래 참조 편향 방지)
        4. Vol scalar 적용 (변동성 기반 포지션 사이징)
        5. ShortMode 처리
        6. Entry/Exit 시그널 생성

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: kama, close, atr, vol_scalar
        config: KAMA 설정

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    if config is None:
        config = KAMAConfig()

    # 입력 검증
    required_cols = {"kama", "close", "atr", "vol_scalar"}
    if config.short_mode == ShortMode.HEDGE_ONLY:
        required_cols.add("drawdown")
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # ================================================================
    # 1. KAMA 방향 및 ATR 필터
    # ================================================================
    kama: pd.Series = df["kama"]  # type: ignore[assignment]
    close: pd.Series = df["close"]  # type: ignore[assignment]
    atr: pd.Series = df["atr"]  # type: ignore[assignment]
    vol_scalar: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    # Shift(1) 적용 — 미래 참조 편향 방지
    kama_prev: pd.Series = kama.shift(1)  # type: ignore[assignment]
    kama_prev2: pd.Series = kama.shift(2)  # type: ignore[assignment]
    close_prev: pd.Series = close.shift(1)  # type: ignore[assignment]
    atr_prev: pd.Series = atr.shift(1)  # type: ignore[assignment]
    vol_scalar_prev: pd.Series = vol_scalar.shift(1)  # type: ignore[assignment]

    # ================================================================
    # 2. 추세 방향 판단
    # ================================================================
    kama_rising = kama_prev > kama_prev2
    kama_falling = kama_prev < kama_prev2

    # ================================================================
    # 3. ATR 필터 적용 진입 시그널
    # ================================================================
    long_signal = (close_prev > kama_prev + config.atr_multiplier * atr_prev) & kama_rising
    short_signal = (close_prev < kama_prev - config.atr_multiplier * atr_prev) & kama_falling

    # ================================================================
    # 4. Direction & Strength
    # ================================================================
    direction_arr = np.where(long_signal, 1, np.where(short_signal, -1, 0))
    direction = pd.Series(direction_arr, index=df.index, name="direction", dtype=int)

    # strength = direction * vol_scalar (shift 이미 적용됨)
    raw_strength: pd.Series = direction * vol_scalar_prev  # type: ignore[assignment]
    strength = pd.Series(
        raw_strength.fillna(0),
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
            "KAMA Signals | Total: %d, Long: %d (%.1f%%), Short: %d (%.1f%%)",
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
