"""Capitulation Wick Reversal 시그널 생성.

3중 필터(ATR spike + Volume surge + Wick ratio)로 capitulation/euphoria 감지.

Signal Formula:
    1. Shift(1) on atr_ratio, vol_ratio, lower_wick_ratio, upper_wick_ratio,
       close_position, vol_scalar
    2. ATR spike: atr_ratio > atr_spike_threshold
    3. Vol surge: vol_ratio > vol_surge_threshold
    4. Capitulation (long): spike + surge + lower_wick > threshold + close near low
    5. Euphoria (short): spike + surge + upper_wick > threshold + close near high
    6. Confirmation: wait N bars after raw signal
    7. Exit: timeout

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.cap_wick_rev.config import CapWickRevConfig, ShortMode
from src.strategy.types import Direction, StrategySignals

# ---- Regime awareness constants ----
_REGIME_VOL_RATIO_HIGH = 2.0
_REGIME_VOL_RATIO_LOW = 0.5
_REGIME_DAMPENING = 0.5
_REGIME_MIN_PERIODS = 30


def generate_signals(
    df: pd.DataFrame,
    config: CapWickRevConfig | None = None,
) -> StrategySignals:
    """Capitulation Wick Reversal 시그널 생성.

    3중 필터로 capitulation/euphoria를 감지하고 confirmation 후 진입합니다.

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
        config: 전략 설정. None이면 기본 설정 사용.

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    if config is None:
        config = CapWickRevConfig()

    required_cols = {
        "atr_ratio",
        "vol_ratio",
        "lower_wick_ratio",
        "upper_wick_ratio",
        "close_position",
        "vol_scalar",
    }
    if config.short_mode == ShortMode.HEDGE_ONLY:
        required_cols.add("drawdown")

    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 1. Shift(1) 적용: 전봉 기준 시그널
    atr_ratio_prev: pd.Series = df["atr_ratio"].shift(1)  # type: ignore[assignment]
    vol_ratio_prev: pd.Series = df["vol_ratio"].shift(1)  # type: ignore[assignment]
    lower_wick_prev: pd.Series = df["lower_wick_ratio"].shift(1)  # type: ignore[assignment]
    upper_wick_prev: pd.Series = df["upper_wick_ratio"].shift(1)  # type: ignore[assignment]
    close_pos_prev: pd.Series = df["close_position"].shift(1)  # type: ignore[assignment]
    vol_scalar_prev: pd.Series = df["vol_scalar"].shift(1)  # type: ignore[assignment]

    # 2. Triple filter: ATR spike + Volume surge
    atr_spike = atr_ratio_prev > config.atr_spike_threshold
    vol_surge = vol_ratio_prev > config.vol_surge_threshold

    # 3. Capitulation detection (long signal):
    #    ATR spike + Vol surge + large lower wick + close near low
    capitulation = (
        atr_spike
        & vol_surge
        & (lower_wick_prev > config.wick_ratio_threshold)
        & (close_pos_prev < config.close_position_threshold)
    )

    # 4. Euphoria detection (short signal):
    #    ATR spike + Vol surge + large upper wick + close near high
    euphoria = (
        atr_spike
        & vol_surge
        & (upper_wick_prev > config.wick_ratio_threshold)
        & (close_pos_prev > (1.0 - config.close_position_threshold))
    )

    # 5. Raw entry direction
    raw_entry = pd.Series(
        np.where(capitulation, 1, np.where(euphoria, -1, 0)),
        index=df.index,
    )

    # 6. Confirmation: shift the raw signal by confirmation_bars
    #    This delays entry by N bars after the capitulation/euphoria event
    if config.confirmation_bars > 0:
        raw_entry = raw_entry.shift(config.confirmation_bars).fillna(0).astype(int)

    # 7. Build position: hold entry direction, release to 0 on timeout
    #    Entry -> direction, 0 -> NaN (hold via ffill), new entry overrides
    raw_direction = pd.Series(
        np.where(raw_entry != 0, raw_entry, np.nan),
        index=df.index,
    )
    position = raw_direction.ffill().fillna(0).astype(int)

    # 8. Timeout: count consecutive bars in same direction
    direction_change = position != position.shift(1)
    direction_group = direction_change.cumsum()
    bars_in_position = direction_group.groupby(direction_group).cumcount()
    timed_out = (bars_in_position >= config.exit_timeout_bars) & (position != 0)
    position = position.where(~timed_out, 0)

    # 9. Strength = position * vol_scalar
    strength_raw = position * vol_scalar_prev.fillna(0)

    # 10. Direction normalization
    direction = pd.Series(
        np.sign(strength_raw).fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    strength = pd.Series(
        strength_raw.fillna(0),
        index=df.index,
        name="strength",
    )

    # 11. Regime awareness: dampen signals in extreme vol environments
    if "realized_vol" in df.columns:
        rvol_prev: pd.Series = df["realized_vol"].shift(1)  # type: ignore[assignment]
        rvol_med: pd.Series = rvol_prev.expanding(  # type: ignore[assignment]
            min_periods=_REGIME_MIN_PERIODS,
        ).median()
        rvol_ratio = rvol_prev / rvol_med.clip(lower=1e-10)
        extreme_regime = (rvol_ratio > _REGIME_VOL_RATIO_HIGH) | (
            rvol_ratio < _REGIME_VOL_RATIO_LOW
        )
        strength = strength.where(~extreme_regime, strength * _REGIME_DAMPENING)

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
