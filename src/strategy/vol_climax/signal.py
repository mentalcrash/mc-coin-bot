"""Volume Climax Reversal Signal Generator.

극단적 거래량 스파이크에서 반전 시그널 생성.

Signal Formula:
    1. Shift(1) 적용: volume_zscore, price_direction, close_position, divergence, vol_scalar
    2. Climax = vol_zscore_prev > climax_threshold
    3. Bullish reversal: climax & price_down & close_near_low (capitulation)
    4. Bearish reversal: climax & price_up & close_near_high (euphoria)
    5. Divergence boost: strength * divergence_boost
    6. Exit: vol_zscore_prev < exit_vol_zscore OR timeout

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.types import Direction, StrategySignals
from src.strategy.vol_climax.config import ShortMode, VolClimaxConfig


def generate_signals(
    df: pd.DataFrame,
    config: VolClimaxConfig | None = None,
) -> StrategySignals:
    """Volume Climax Reversal 시그널 생성.

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
        config: Volume Climax 설정. None이면 기본 설정 사용.

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    if config is None:
        config = VolClimaxConfig()

    required_cols = {
        "volume_zscore",
        "price_direction",
        "close_position",
        "divergence",
        "vol_scalar",
    }
    if config.short_mode == ShortMode.HEDGE_ONLY:
        required_cols.add("drawdown")

    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 1. Shift(1) 적용: 전봉 기준 시그널
    vol_zscore_prev: pd.Series = df["volume_zscore"].shift(1)  # type: ignore[assignment]
    price_dir_prev: pd.Series = df["price_direction"].shift(1)  # type: ignore[assignment]
    close_pos_prev: pd.Series = df["close_position"].shift(1)  # type: ignore[assignment]
    divergence_shifted = df["divergence"].astype("boolean").shift(1)
    divergence_prev = divergence_shifted.fillna(False).to_numpy(dtype=bool, na_value=False)
    vol_scalar_prev: pd.Series = df["vol_scalar"].shift(1)  # type: ignore[assignment]

    # 2. Climax detection
    climax = vol_zscore_prev > config.climax_threshold

    # 3. Bullish reversal: climax + price down + close near low (capitulation)
    bullish_reversal = (
        climax & (price_dir_prev < 0) & (close_pos_prev < config.close_position_threshold)
    )

    # 4. Bearish reversal: climax + price up + close near high (euphoria)
    bearish_reversal = (
        climax & (price_dir_prev > 0) & (close_pos_prev > (1.0 - config.close_position_threshold))
    )

    # 5. Raw direction from reversal signals
    raw_entry_direction = pd.Series(
        np.where(
            bullish_reversal,
            1,
            np.where(bearish_reversal, -1, np.nan),
        ),
        index=df.index,
    )

    # 6. Exit conditions: vol_zscore drops below exit threshold
    exit_cond = vol_zscore_prev < config.exit_vol_zscore

    # Apply exit: force to neutral when exit condition met
    raw_direction = pd.Series(
        np.where(exit_cond, 0, raw_entry_direction),
        index=df.index,
    )

    # Forward-fill to hold position (NaN = maintain previous)
    raw_direction = raw_direction.ffill().fillna(0).astype(int)

    # 7. Timeout: count consecutive bars in same direction
    direction_change = raw_direction != raw_direction.shift(1)
    direction_group = direction_change.cumsum()
    bars_in_position = direction_group.groupby(direction_group).cumcount()

    # Apply timeout: force to neutral after exit_timeout_bars
    timed_out = (bars_in_position >= config.exit_timeout_bars) & (raw_direction != 0)
    raw_direction = raw_direction.where(~timed_out, 0)

    # 8. Base strength = direction * vol_scalar
    strength_raw = raw_direction * vol_scalar_prev

    # 9. Divergence boost: amplify conviction when OBV diverges from price
    has_divergence = divergence_prev & (raw_direction != 0)
    strength_raw = strength_raw.where(
        ~has_divergence,
        strength_raw * config.divergence_boost,
    )

    # 10. Direction 정규화
    direction = pd.Series(
        np.sign(strength_raw).fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    # 11. 강도 계산
    strength = pd.Series(
        strength_raw.fillna(0),
        index=df.index,
        name="strength",
    )

    # 12. 숏 모드에 따른 시그널 처리
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

        hedge_bars = int(hedge_active.sum())
        if hedge_bars > 0:
            logger.info(
                "Hedge Mode | Active: {} bars ({:.1f}%), Threshold: {:.1f}%",
                hedge_bars,
                hedge_bars / len(hedge_active) * 100,
                config.hedge_threshold * 100,
            )

    # 13. 진입/청산 시그널
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
