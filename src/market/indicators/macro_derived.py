"""Macro-derived indicators (credit spread, risk appetite, cross-asset correlation).

매크로/cross-asset 데이터 기반 파생 지표 함수.
모든 함수는 stateless, vectorized이며 ``pd.Series`` in → ``pd.Series`` out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def credit_spread_proxy(
    hyg_close: pd.Series,
    tlt_close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Credit Spread Proxy — HYG vs TLT 수익률 차이의 rolling 평균.

    높은 값: 크레딧 스프레드 확대 (risk-off).
    낮은 값: 크레딧 스프레드 축소 (risk-on).

    Args:
        hyg_close: HYG (하이일드 채권) 종가 시리즈.
        tlt_close: TLT (장기 국채) 종가 시리즈.
        window: Rolling 평균 윈도우 (기본 20일).

    Returns:
        Credit spread proxy 시리즈.
    """
    hyg_ret = hyg_close.pct_change()
    tlt_ret = tlt_close.pct_change()
    # TLT 수익률 - HYG 수익률: HYG 부진 시 스프레드 확대
    spread: pd.Series = tlt_ret - hyg_ret  # type: ignore[assignment]
    result: pd.Series = spread.rolling(  # type: ignore[assignment]
        window=window, min_periods=window
    ).mean()
    return result


def risk_appetite_index(
    spy_close: pd.Series,
    tlt_close: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Risk Appetite Index — rolling_zscore(SPY_ret - TLT_ret).

    양수: 위험자산 선호 (risk-on).
    음수: 안전자산 선호 (risk-off).

    Args:
        spy_close: SPY (S&P500) 종가 시리즈.
        tlt_close: TLT (장기 국채) 종가 시리즈.
        window: Z-score rolling 윈도우 (기본 60일).

    Returns:
        Risk appetite z-score 시리즈.
    """
    spy_ret = spy_close.pct_change()
    tlt_ret = tlt_close.pct_change()
    diff: pd.Series = spy_ret - tlt_ret  # type: ignore[assignment]

    rolling_mean = diff.rolling(window=window, min_periods=window).mean()
    rolling_std = diff.rolling(window=window, min_periods=window).std()
    rolling_std_safe = rolling_std.replace(0, np.nan)
    result: pd.Series = (diff - rolling_mean) / rolling_std_safe  # type: ignore[assignment]
    return result


def btc_spy_correlation(
    btc_close: pd.Series,
    spy_close: pd.Series,
    window: int = 21,
) -> pd.Series:
    """BTC-SPY Rolling Correlation — BTC와 S&P500의 수익률 상관.

    높은 상관: 매크로 리스크 동조화 (전통 분산효과 감소).
    낮은/음의 상관: BTC 독립적 (분산 투자 효과).

    Args:
        btc_close: BTC 종가 시리즈.
        spy_close: SPY 종가 시리즈.
        window: Rolling 상관 윈도우 (기본 21일).

    Returns:
        Rolling 상관계수 시리즈 (-1 ~ +1).
    """
    btc_ret = btc_close.pct_change()
    spy_ret = spy_close.pct_change()
    result: pd.Series = btc_ret.rolling(window).corr(spy_ret)  # type: ignore[assignment]
    return result
