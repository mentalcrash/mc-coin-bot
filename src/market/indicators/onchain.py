"""On-chain indicators (Puell Multiple, NVT, MVRV, Exchange Flow, etc.).

온체인 데이터 기반 지표 함수.
모든 함수는 stateless, vectorized이며 ``pd.Series`` in → ``pd.Series`` out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def puell_multiple(
    daily_miner_revenue: pd.Series,
    ma_window: int = 365,
) -> pd.Series:
    """Puell Multiple — 일일 채굴 수익 / 365일 이동평균.

    높은 값(>4)은 과열, 낮은 값(<0.5)은 저평가 구간.

    Args:
        daily_miner_revenue: 일일 채굴 수익 시리즈 (USD).
        ma_window: 이동평균 윈도우 (기본 365일).

    Returns:
        Puell Multiple 시리즈.
    """
    ma: pd.Series = daily_miner_revenue.rolling(  # type: ignore[assignment]
        window=ma_window, min_periods=ma_window
    ).mean()
    ma_safe = ma.replace(0, np.nan)
    result: pd.Series = daily_miner_revenue / ma_safe  # type: ignore[assignment]
    return result


def nvt_signal(
    market_cap: pd.Series,
    tx_volume: pd.Series,
    ma_window: int = 90,
) -> pd.Series:
    """NVT Signal — Market Cap / MA(Transaction Volume).

    NVT Ratio의 smoothed 버전. 높은 값은 네트워크 가치 대비
    거래량 부족 (과대평가), 낮은 값은 저평가.

    Args:
        market_cap: 시가총액 시리즈 (USD).
        tx_volume: 온체인 거래량 시리즈 (USD).
        ma_window: 거래량 smoothing 윈도우 (기본 90일).

    Returns:
        NVT Signal 시리즈.
    """
    tx_vol_ma: pd.Series = tx_volume.rolling(  # type: ignore[assignment]
        window=ma_window, min_periods=ma_window
    ).mean()
    tx_vol_safe = tx_vol_ma.replace(0, np.nan)
    result: pd.Series = market_cap / tx_vol_safe  # type: ignore[assignment]
    return result


def mvrv_zscore(
    mvrv: pd.Series,
    window: int = 365,
) -> pd.Series:
    """MVRV Z-Score — (MVRV - MA) / StdDev.

    MVRV의 통계적 극단값 탐지. Z>7 시장 과열, Z<-0.3 저점 구간.

    Args:
        mvrv: MVRV 비율 시리즈.
        window: Rolling 윈도우 (기본 365일).

    Returns:
        MVRV Z-Score 시리즈.
    """
    rolling_mean = mvrv.rolling(window=window, min_periods=window).mean()
    rolling_std = mvrv.rolling(window=window, min_periods=window).std()
    rolling_std_safe = rolling_std.replace(0, np.nan)
    result: pd.Series = (mvrv - rolling_mean) / rolling_std_safe  # type: ignore[assignment]
    return result


def exchange_flow_net_zscore(
    flow_in: pd.Series,
    flow_out: pd.Series,
    window: int = 30,
) -> pd.Series:
    """Exchange Net Flow Z-Score — rolling_zscore(flow_in - flow_out).

    양수 z-score: 비정상적 입금 증가 (매도 압력).
    음수 z-score: 비정상적 출금 증가 (축적).

    Args:
        flow_in: 거래소 입금량 시리즈 (USD).
        flow_out: 거래소 출금량 시리즈 (USD).
        window: Z-score rolling 윈도우 (기본 30일).

    Returns:
        Net flow z-score 시리즈.
    """
    net_flow: pd.Series = flow_in - flow_out  # type: ignore[assignment]
    rolling_mean = net_flow.rolling(window=window, min_periods=window).mean()
    rolling_std = net_flow.rolling(window=window, min_periods=window).std()
    rolling_std_safe = rolling_std.replace(0, np.nan)
    result: pd.Series = (net_flow - rolling_mean) / rolling_std_safe  # type: ignore[assignment]
    return result


def stablecoin_supply_ratio(
    btc_market_cap: pd.Series,
    stablecoin_supply: pd.Series,
) -> pd.Series:
    """Stablecoin Supply Ratio (SSR) — BTC Market Cap / Stablecoin Supply.

    높은 SSR: 스테이블코인 대비 BTC 시총 크다 (매수 여력 부족).
    낮은 SSR: 스테이블코인 풍부 (잠재 매수력).

    Args:
        btc_market_cap: BTC 시가총액 시리즈 (USD).
        stablecoin_supply: 스테이블코인 총 발행량 시리즈 (USD).

    Returns:
        SSR 시리즈.
    """
    supply_safe = stablecoin_supply.replace(0, np.nan)
    result: pd.Series = btc_market_cap / supply_safe  # type: ignore[assignment]
    return result


def tvl_stablecoin_ratio(
    total_tvl: pd.Series,
    stablecoin_supply: pd.Series,
) -> pd.Series:
    """TVL / Stablecoin Supply Ratio — DeFi 활용도 지표.

    높은 비율: DeFi에 적극 투입 (risk-on).
    낮은 비율: 유동성 대기 (risk-off).

    Args:
        total_tvl: DeFi 총 TVL 시리즈 (USD).
        stablecoin_supply: 스테이블코인 총 발행량 시리즈 (USD).

    Returns:
        TVL/Stablecoin 비율 시리즈.
    """
    supply_safe = stablecoin_supply.replace(0, np.nan)
    result: pd.Series = total_tvl / supply_safe  # type: ignore[assignment]
    return result
