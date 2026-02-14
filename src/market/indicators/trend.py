"""Trend and moving average indicators."""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ATR (Average True Range) — Wilder's smoothing.

    Args:
        high: 고가 시리즈.
        low: 저가 시리즈.
        close: 종가 시리즈.
        period: 계산 기간.

    Returns:
        ATR 시리즈.
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    result = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return pd.Series(result, index=close.index, name="atr")


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ADX (Average Directional Index).

    Args:
        high: 고가 시리즈.
        low: 저가 시리즈.
        close: 종가 시리즈.
        period: 계산 기간.

    Returns:
        ADX 시리즈 (0-100).
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0),
        index=high.index,
    )

    atr_val = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di: pd.Series = 100 * (
        plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_val
    )  # type: ignore[assignment]
    minus_di: pd.Series = 100 * (
        minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_val
    )  # type: ignore[assignment]

    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100 * (di_diff / di_sum.replace(0, np.nan))

    result = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return pd.Series(result, index=close.index, name="adx")


def sma(close: pd.Series, period: int) -> pd.Series:
    """SMA (Simple Moving Average).

    Args:
        close: 종가 시리즈.
        period: 이동평균 기간.

    Returns:
        SMA 시리즈.
    """
    result: pd.Series = close.rolling(window=period).mean()  # type: ignore[assignment]
    return result


def ema(close: pd.Series, span: int) -> pd.Series:
    """EMA (Exponential Moving Average).

    Args:
        close: 종가 시리즈.
        span: EMA span.

    Returns:
        EMA 시리즈.
    """
    result: pd.Series = close.ewm(span=span, adjust=False).mean()  # type: ignore[assignment]
    return result


def efficiency_ratio(close: pd.Series, period: int) -> pd.Series:
    """Kaufman Efficiency Ratio (ER).

    ER = |direction| / volatility, where
    direction = close[t] - close[t-period]
    volatility = sum(|close[i] - close[i-1]|) over period

    Args:
        close: 종가 시리즈.
        period: 룩백 기간.

    Returns:
        Efficiency Ratio 시리즈 (0~1).
    """
    direction = (close - close.shift(period)).abs()
    volatility: pd.Series = (
        close.diff()
        .abs()
        .rolling(  # type: ignore[assignment]
            period, min_periods=period
        )
        .sum()
    )
    er = direction / volatility.replace(0, np.nan)
    return pd.Series(er.fillna(0), index=close.index, name="efficiency_ratio")


@numba.njit  # type: ignore[misc]
def _compute_kama_numba(
    close_arr: npt.NDArray[np.float64],
    sc_arr: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """KAMA recursive calculation (numba-optimized)."""
    n = len(close_arr)
    kama_arr = np.empty(n)
    kama_arr[0] = close_arr[0]
    for i in range(1, n):
        if np.isnan(sc_arr[i]) or np.isnan(close_arr[i]):
            kama_arr[i] = kama_arr[i - 1]
        else:
            kama_arr[i] = kama_arr[i - 1] + sc_arr[i] * (close_arr[i] - kama_arr[i - 1])
    return kama_arr


def kama(
    close: pd.Series,
    er_lookback: int = 10,
    fast_period: int = 2,
    slow_period: int = 30,
) -> pd.Series:
    """Kaufman Adaptive Moving Average (KAMA).

    Efficiency Ratio(ER)를 기반으로 시장 상태에 적응하는 이동평균.
    추세가 강할수록(ER -> 1) 빠른 EMA에, 횡보일수록(ER -> 0) 느린 EMA에 수렴.

    Args:
        close: 종가 시리즈.
        er_lookback: Efficiency Ratio 룩백 기간.
        fast_period: 빠른 SC 기간.
        slow_period: 느린 SC 기간.

    Returns:
        KAMA 시리즈.
    """
    er = efficiency_ratio(close, er_lookback)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama_values = _compute_kama_numba(
        close.to_numpy().astype(np.float64),
        sc.to_numpy().astype(np.float64),
    )
    return pd.Series(kama_values, index=close.index, name="kama")
