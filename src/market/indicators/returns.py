"""Return calculation indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    """로그 수익률: ln(P_t / P_{t-1}).

    Args:
        close: 종가 시리즈.

    Returns:
        로그 수익률 시리즈 (첫 값은 NaN).

    Raises:
        ValueError: 빈 시리즈.
    """
    if len(close) == 0:
        msg = "Empty Series provided"
        raise ValueError(msg)
    price_ratio = close / close.shift(1)
    return pd.Series(np.log(price_ratio), index=close.index, name="returns")


def simple_returns(close: pd.Series) -> pd.Series:
    """단순 수익률: (P_t - P_{t-1}) / P_{t-1}.

    Args:
        close: 종가 시리즈.

    Returns:
        단순 수익률 시리즈 (첫 값은 NaN).

    Raises:
        ValueError: 빈 시리즈.
    """
    if len(close) == 0:
        msg = "Empty Series provided"
        raise ValueError(msg)
    return close.pct_change()


def rolling_return(
    close: pd.Series,
    period: int,
    use_log: bool = True,
) -> pd.Series:
    """Rolling return over *period* bars.

    Args:
        close: 종가 시리즈.
        period: 수익률 계산 기간.
        use_log: True면 로그 수익률, False면 단순 수익률.

    Returns:
        Rolling return 시리즈.

    Raises:
        ValueError: 빈 시리즈.
    """
    if len(close) == 0:
        msg = "Empty Series provided"
        raise ValueError(msg)
    if use_log:
        price_ratio = close / close.shift(period)
        return pd.Series(np.log(price_ratio), index=close.index, name="rolling_return")
    return pd.Series(close.pct_change(period), index=close.index, name="rolling_return")
