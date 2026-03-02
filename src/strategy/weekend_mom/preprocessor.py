"""Weekend-Momentum 전처리 모듈.

OHLCV 데이터에서 weekend-weighted momentum feature를 계산한다.
주말(토/일) 수익률에 가중치를 부여하여 momentum 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.market.indicators import (
    atr,
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    from src.strategy.weekend_mom.config import WeekendMomConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})

# Saturday=5 in pandas dayofweek (0=Mon, 6=Sun)
_WEEKEND_START_DOW = 5


def _weekend_weighted_returns(
    returns: pd.Series,
    index: pd.DatetimeIndex,
    weekend_boost: float,
) -> pd.Series:
    """Weekend return에 가중치를 부여한 수익률 시리즈.

    Args:
        returns: Log return 시리즈.
        index: DatetimeIndex (요일 추출용).
        weekend_boost: 주말 수익률 가중치 배수 (1.0 = 균등).

    Returns:
        Weekend-weighted returns.
    """
    # dayofweek: 0=Mon, ..., 5=Sat, 6=Sun
    day_of_week = index.dayofweek  # type: ignore[union-attr]
    is_weekend = (day_of_week >= _WEEKEND_START_DOW).astype(float)
    weights = np.where(is_weekend, weekend_boost, 1.0)
    return returns * weights


def _rolling_weighted_mom(
    weighted_returns: pd.Series,
    lookback: int,
) -> pd.Series:
    """Rolling sum of weekend-weighted returns (momentum).

    Args:
        weighted_returns: Weekend-weighted return 시리즈.
        lookback: Rolling window (bars).

    Returns:
        Rolling momentum 시리즈.
    """
    result: pd.Series = weighted_returns.rolling(window=lookback, min_periods=lookback).sum()  # type: ignore[assignment]
    return result


def preprocess(df: pd.DataFrame, config: WeekendMomConfig) -> pd.DataFrame:
    """Weekend-Momentum feature 계산.

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: 전략 설정

    Returns:
        feature가 추가된 새 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    df = df.copy()

    close: pd.Series = df["close"]  # type: ignore[assignment]
    high: pd.Series = df["high"]  # type: ignore[assignment]
    low: pd.Series = df["low"]  # type: ignore[assignment]

    # --- Returns ---
    returns = log_returns(close)
    df["returns"] = returns

    # --- Realized Volatility ---
    realized_vol = realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    df["realized_vol"] = realized_vol

    # --- Vol Scalar ---
    df["vol_scalar"] = volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # --- Weekend-Weighted Returns ---
    index = pd.DatetimeIndex(df.index)
    weighted_returns = _weekend_weighted_returns(returns, index, config.weekend_boost)
    df["weighted_returns"] = weighted_returns

    # --- Weekend indicator (for diagnostics / tests) ---
    df["is_weekend"] = (index.dayofweek >= _WEEKEND_START_DOW).astype(float)  # type: ignore[union-attr]

    # --- Multi-Scale Weekend Momentum ---
    df["fast_mom"] = _rolling_weighted_mom(weighted_returns, config.fast_lookback)
    df["slow_mom"] = _rolling_weighted_mom(weighted_returns, config.slow_lookback)

    # --- Combined Momentum Score ---
    # fast/slow 합산을 slow window std로 정규화하여 비교 가능하게 만듦
    combined_raw = df["fast_mom"] + df["slow_mom"]
    combined_std = combined_raw.rolling(
        window=config.slow_lookback, min_periods=config.slow_lookback
    ).std()
    combined_mean = combined_raw.rolling(
        window=config.slow_lookback, min_periods=config.slow_lookback
    ).mean()
    df["mom_score"] = (combined_raw - combined_mean) / combined_std.clip(lower=1e-10)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    return df
