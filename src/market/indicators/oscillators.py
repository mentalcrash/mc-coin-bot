"""Oscillator indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Relative Strength Index) — Wilder's smoothing.

    Args:
        close: 종가 시리즈.
        period: 계산 기간.

    Returns:
        RSI 시리즈 (0-100).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    # 100 * gain / (gain + loss) — algebraically equivalent to 100 - 100/(1+RS)
    # but handles zero-loss (RSI=100) and zero-gain (RSI=0) correctly.
    denom = (avg_gain + avg_loss).replace(0, np.nan)
    result: pd.Series = 100 * avg_gain / denom  # type: ignore[assignment]
    return pd.Series(result, index=close.index, name="rsi")


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic (%K, %D).

    Args:
        high: 고가 시리즈.
        low: 저가 시리즈.
        close: 종가 시리즈.
        k_period: %K 기간.
        d_period: %D 기간.

    Returns:
        (%K, %D) 튜플.
    """
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    denom = (highest - lowest).replace(0, np.nan)
    k: pd.Series = 100 * (close - lowest) / denom  # type: ignore[assignment]
    d: pd.Series = k.rolling(d_period).mean()  # type: ignore[assignment]
    return k, d


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Williams %R (-100 to 0).

    Args:
        high: 고가 시리즈.
        low: 저가 시리즈.
        close: 종가 시리즈.
        period: 계산 기간.

    Returns:
        Williams %R 시리즈.
    """
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    denom = (highest - lowest).replace(0, np.nan)
    wr: pd.Series = -100 * (highest - close) / denom  # type: ignore[assignment]
    return wr


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """CCI (Commodity Channel Index).

    Args:
        high: 고가 시리즈.
        low: 저가 시리즈.
        close: 종가 시리즈.
        period: 계산 기간.

    Returns:
        CCI 시리즈.
    """
    tp: pd.Series = (high + low + close) / 3  # type: ignore[assignment]
    tp_sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(
        lambda x: np.abs(x - x.mean()).mean(),  # type: ignore[reportUnknownLambdaType]
        raw=True,
    )
    result: pd.Series = (tp - tp_sma) / (0.015 * mad.replace(0, np.nan))  # type: ignore[assignment]
    return result


def roc(close: pd.Series, period: int) -> pd.Series:
    """ROC (Rate of Change).

    Args:
        close: 종가 시리즈.
        period: 계산 기간.

    Returns:
        ROC 시리즈.
    """
    return close.pct_change(period)


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD (Moving Average Convergence Divergence).

    Args:
        close: 종가 시리즈.
        fast: 빠른 EMA 기간.
        slow: 느린 EMA 기간.
        signal: 시그널 EMA 기간.

    Returns:
        (MACD line, Signal line, Histogram) 튜플.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line: pd.Series = ema_fast - ema_slow  # type: ignore[assignment]
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram: pd.Series = macd_line - signal_line  # type: ignore[assignment]
    return macd_line, signal_line, histogram


def momentum(close: pd.Series, period: int) -> pd.Series:
    """Price momentum: close[t] - close[t-period].

    Args:
        close: 종가 시리즈.
        period: 모멘텀 계산 기간.

    Returns:
        Momentum 시리즈.
    """
    return pd.Series(close.diff(period), index=close.index, name="momentum")
