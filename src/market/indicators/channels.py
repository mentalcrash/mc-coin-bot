"""Channel indicators (bands, channels)."""

from __future__ import annotations

import pandas as pd

from .trend import atr as _atr


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """볼린저밴드: (upper, middle, lower).

    Args:
        close: 종가 시리즈.
        period: SMA 기간.
        std_dev: 표준편차 배수.

    Returns:
        (bb_upper, bb_middle, bb_lower) 튜플.
    """
    bb_middle = close.rolling(window=period, min_periods=period).mean()
    rolling_std = close.rolling(window=period, min_periods=period).std()

    bb_upper: pd.Series = bb_middle + std_dev * rolling_std  # type: ignore[assignment]
    bb_lower: pd.Series = bb_middle - std_dev * rolling_std  # type: ignore[assignment]

    return (
        pd.Series(bb_upper, index=close.index, name="bb_upper"),
        pd.Series(bb_middle, index=close.index, name="bb_middle"),
        pd.Series(bb_lower, index=close.index, name="bb_lower"),
    )


def donchian_channel(
    high: pd.Series,
    low: pd.Series,
    period: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian Channel: (upper, middle, lower).

    Args:
        high: 고가 시리즈.
        low: 저가 시리즈.
        period: Lookback 기간.

    Returns:
        (upper, middle, lower) 튜플.
    """
    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()
    middle: pd.Series = (upper + lower) / 2  # type: ignore[assignment]
    return (
        pd.Series(upper, index=high.index, name="dc_upper"),
        pd.Series(middle, index=high.index, name="dc_middle"),
        pd.Series(lower, index=high.index, name="dc_lower"),
    )


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 1.5,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channels: (upper, middle, lower).

    Args:
        high: 고가 시리즈.
        low: 저가 시리즈.
        close: 종가 시리즈.
        ema_period: EMA 기간.
        atr_period: ATR 기간.
        multiplier: ATR 배수.

    Returns:
        (kc_upper, kc_middle, kc_lower) 튜플.
    """
    kc_middle = close.ewm(span=ema_period, adjust=False).mean()
    atr_val = _atr(high, low, close, period=atr_period)
    kc_upper: pd.Series = kc_middle + multiplier * atr_val  # type: ignore[assignment]
    kc_lower: pd.Series = kc_middle - multiplier * atr_val  # type: ignore[assignment]
    return (
        pd.Series(kc_upper, index=close.index, name="kc_upper"),
        pd.Series(kc_middle, index=close.index, name="kc_middle"),
        pd.Series(kc_lower, index=close.index, name="kc_lower"),
    )
