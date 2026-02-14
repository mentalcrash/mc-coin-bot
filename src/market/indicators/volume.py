"""Volume-based indicators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """OBV (On Balance Volume).

    Args:
        close: 종가 시리즈.
        volume: 거래량 시리즈.

    Returns:
        OBV 시리즈.
    """
    direction = np.sign(close.diff())
    result: pd.Series = (volume * direction).cumsum()  # type: ignore[assignment]
    return result


def volume_weighted_returns(
    returns: pd.Series,
    volume: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """거래량 가중 수익률 (로그 스케일링).

    Args:
        returns: 수익률 시리즈.
        volume: 거래량 시리즈.
        window: Rolling 윈도우 크기.
        min_periods: 최소 관측치 수.

    Returns:
        거래량 가중 수익률 시리즈.
    """
    if min_periods is None:
        min_periods = window
    log_volume = np.log1p(volume)
    weighted: pd.Series = (  # type: ignore[assignment]
        (returns * log_volume).rolling(window=window, min_periods=min_periods).sum()
    )
    total_log_vol: pd.Series = log_volume.rolling(  # type: ignore[assignment]
        window=window, min_periods=min_periods
    ).sum()
    total_log_vol_safe = total_log_vol.replace(0, np.nan)
    return weighted / total_log_vol_safe


def chaikin_money_flow(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Chaikin Money Flow.

    Args:
        high: 고가 시리즈.
        low: 저가 시리즈.
        close: 종가 시리즈.
        volume: 거래량 시리즈.
        period: 계산 기간.

    Returns:
        CMF 시리즈.
    """
    hl_range = (high - low).replace(0, np.nan)
    mfm: pd.Series = ((close - low) - (high - close)) / hl_range  # type: ignore[assignment]
    mfv: pd.Series = mfm * volume  # type: ignore[assignment]
    vol_sum: pd.Series = volume.rolling(period).sum()  # type: ignore[assignment]
    vol_sum = vol_sum.replace(0, np.nan)
    cmf: pd.Series = mfv.rolling(period).sum() / vol_sum  # type: ignore[assignment]
    return cmf


def volume_macd(
    volume: pd.Series,
    fast: int = 12,
    slow: int = 26,
) -> pd.Series:
    """Volume MACD (EMA difference).

    Args:
        volume: 거래량 시리즈.
        fast: 빠른 EMA 기간.
        slow: 느린 EMA 기간.

    Returns:
        Volume MACD 시리즈.
    """
    v_macd: pd.Series = (
        volume.ewm(span=fast, adjust=False).mean() - volume.ewm(span=slow, adjust=False).mean()
    )  # type: ignore[assignment]
    return v_macd
