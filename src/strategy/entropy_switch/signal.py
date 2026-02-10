"""Entropy Regime Switch Signal Generator.

Shannon Entropy로 regime 분류 후 trend-following/flat 전환.

Signal Formula:
    1. Shift(1) on entropy, mom_direction, adx, vol_scalar
    2. Low entropy + positive momentum → LONG
    3. Low entropy + negative momentum → SHORT (HEDGE_ONLY)
    4. High entropy → FLAT (no signal)
    5. Middle zone → NEUTRAL (hold existing, no new entry)
    6. strength = direction * vol_scalar

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.entropy_switch.config import EntropySwitchConfig, ShortMode
from src.strategy.types import Direction, StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: EntropySwitchConfig | None = None,
) -> StrategySignals:
    """Entropy Regime Switch 시그널 생성.

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
        config: Entropy Switch 설정. None이면 기본 설정 사용.

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    if config is None:
        config = EntropySwitchConfig()

    required_cols = {"entropy", "mom_direction", "vol_scalar"}
    if config.use_adx_filter:
        required_cols.add("adx")
    if config.short_mode == ShortMode.HEDGE_ONLY:
        required_cols.add("drawdown")

    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 1. Shift(1) 적용: 전봉 기준 시그널
    entropy_prev: pd.Series = df["entropy"].shift(1)  # type: ignore[assignment]
    mom_dir_prev: pd.Series = df["mom_direction"].shift(1)  # type: ignore[assignment]
    vol_scalar_prev: pd.Series = df["vol_scalar"].shift(1)  # type: ignore[assignment]

    # 2. Entropy regime classification
    low_entropy = entropy_prev < config.entropy_low_threshold
    high_entropy = entropy_prev > config.entropy_high_threshold
    # middle_zone = ~low_entropy & ~high_entropy  (implicit: direction stays 0)

    # 3. ADX auxiliary filter (optional)
    if config.use_adx_filter:
        adx_prev: pd.Series = df["adx"].shift(1)  # type: ignore[assignment]
        adx_pass = adx_prev > config.adx_threshold
        tradeable = low_entropy & adx_pass
    else:
        tradeable = low_entropy

    # 4. Direction:
    #    Low entropy (+ ADX pass) → follow momentum
    #    High entropy → FLAT (0)
    #    Middle zone → NEUTRAL (0, no new entry)
    direction_raw = pd.Series(
        np.where(
            tradeable,
            mom_dir_prev,
            0,
        ),
        index=df.index,
    )

    # High entropy explicitly forces flat (already 0, but explicit for clarity)
    direction_raw = direction_raw.where(~high_entropy, 0)

    # 5. Strength = direction * vol_scalar
    strength_raw = direction_raw * vol_scalar_prev

    # 6. Direction 정규화
    direction = pd.Series(
        np.sign(strength_raw).fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    # 7. 강도 계산
    strength = pd.Series(
        strength_raw.fillna(0),
        index=df.index,
        name="strength",
    )

    # 8. 숏 모드에 따른 시그널 처리
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

    # 9. 진입/청산 시그널
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
