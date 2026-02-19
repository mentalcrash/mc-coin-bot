"""Session Breakout Signal Generator.

Asian range breakout 방향 추종 (1H).

Signal Formula:
    1. Shift(1) 적용: asian_high, asian_low, range_pctl, vol_scalar, close
    2. squeeze = range_pctl_prev < threshold
    3. Long: is_trade_prev & squeeze & (close_prev > asian_high_prev)
    4. Short: is_trade_prev & squeeze & (close_prev < asian_low_prev)
    5. Exit: is_exit_prev → direction = 0

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.session_breakout.config import SessionBreakoutConfig, ShortMode
from src.strategy.types import Direction, StrategySignals

# ---- Regime awareness constants ----
_REGIME_VOL_RATIO_HIGH = 2.0
_REGIME_VOL_RATIO_LOW = 0.5
_REGIME_DAMPENING = 0.5
_REGIME_MIN_PERIODS = 30

# ---- Turnover control constants ----
_MIN_HOLD_BARS = 4


def generate_signals(
    df: pd.DataFrame,
    config: SessionBreakoutConfig | None = None,
) -> StrategySignals:
    """Session Breakout 시그널 생성.

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
        config: Session Breakout 설정. None이면 기본 설정 사용.

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    if config is None:
        config = SessionBreakoutConfig()

    required_cols = {
        "asian_high",
        "asian_low",
        "range_pctl",
        "vol_scalar",
        "close",
        "is_trade_window",
        "is_exit_hour",
    }
    if config.short_mode == ShortMode.HEDGE_ONLY:
        required_cols.add("drawdown")

    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 1. Shift(1) 적용: 전봉 기준 시그널
    asian_high_prev: pd.Series = df["asian_high"].shift(1)  # type: ignore[assignment]
    asian_low_prev: pd.Series = df["asian_low"].shift(1)  # type: ignore[assignment]
    range_pctl_prev: pd.Series = df["range_pctl"].shift(1)  # type: ignore[assignment]
    vol_scalar_prev: pd.Series = df["vol_scalar"].shift(1)  # type: ignore[assignment]
    close_prev: pd.Series = df["close"].shift(1)  # type: ignore[assignment]
    is_trade_prev = (
        df["is_trade_window"].shift(1).fillna(False).infer_objects(copy=False).astype(bool)
    )
    is_exit_prev = df["is_exit_hour"].shift(1).fillna(False).infer_objects(copy=False).astype(bool)

    # 2. Squeeze 감지: range percentile < threshold
    squeeze = range_pctl_prev < config.range_pctl_threshold

    # 3. Breakout direction
    long_signal = is_trade_prev & squeeze & (close_prev > asian_high_prev)
    short_signal = is_trade_prev & squeeze & (close_prev < asian_low_prev)

    direction_raw = pd.Series(
        np.where(
            is_exit_prev,
            0,
            np.where(long_signal, 1, np.where(short_signal, -1, np.nan)),
        ),
        index=df.index,
    )
    # Turnover control: hold breakout position until exit or reversal
    direction_raw = direction_raw.ffill().fillna(0).astype(int)

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

    # 7. Regime awareness: dampen signals in extreme vol environments
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

        hedge_bars = int(hedge_active.sum())
        if hedge_bars > 0:
            logger.info(
                "Hedge Mode | Active: {} bars ({:.1f}%), Threshold: {:.1f}%",
                hedge_bars,
                hedge_bars / len(hedge_active) * 100,
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
