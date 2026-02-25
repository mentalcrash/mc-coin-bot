"""Market microstructure indicators (VPIN, CVD divergence, liquidation metrics, order flow).

시장 미시구조 데이터 기반 지표 함수.
모든 함수는 stateless, vectorized이며 ``pd.Series`` in → ``pd.Series`` out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def vpin(
    close: pd.Series,
    volume: pd.Series,
    window: int = 50,
) -> pd.Series:
    """Volume-Synchronized Probability of Informed Trading (VPIN).

    Bulk Volume Classification 방식: 종가 변화 방향으로 거래량 분류.
    높은 VPIN: 정보 비대칭 증가 (informed trading 활발).

    Args:
        close: 종가 시리즈.
        volume: 거래량 시리즈.
        window: Rolling 윈도우 (기본 50 bars).

    Returns:
        VPIN 시리즈 (0~1).
    """
    price_change = close.diff()
    # Bulk Volume Classification: 가격 상승→매수, 하락→매도
    buy_vol = volume.where(price_change > 0, 0.0)
    sell_vol = volume.where(price_change < 0, 0.0)
    # 변화 없는 bar는 반반 분배
    unchanged = price_change == 0
    buy_vol = buy_vol + volume.where(unchanged, 0.0) * 0.5
    sell_vol = sell_vol + volume.where(unchanged, 0.0) * 0.5

    abs_imbalance = (buy_vol - sell_vol).abs()
    total_vol: pd.Series = volume.rolling(  # type: ignore[assignment]
        window=window, min_periods=window
    ).sum()
    total_vol_safe = total_vol.replace(0, np.nan)
    rolling_imbalance = abs_imbalance.rolling(window=window, min_periods=window).sum()
    result: pd.Series = rolling_imbalance / total_vol_safe  # type: ignore[assignment]
    return result


def cvd_price_divergence(
    close: pd.Series,
    cvd: pd.Series,
    window: int = 14,
) -> pd.Series:
    """CVD-Price Divergence Score — 가격과 CVD(누적 거래량 델타) 방향 불일치.

    양수: 가격 상승 + CVD 하락 (bearish divergence).
    음수: 가격 하락 + CVD 상승 (bullish divergence).
    0: 방향 일치.

    Args:
        close: 종가 시리즈.
        cvd: 누적 거래량 델타 시리즈.
        window: ROC 계산 윈도우 (기본 14일).

    Returns:
        Divergence score 시리즈 (-1, 0, +1).
    """
    close_safe = close.shift(window).replace(0, np.nan)
    cvd_shift = cvd.shift(window).replace(0, np.nan)

    price_roc = (close - close.shift(window)) / close_safe
    cvd_roc = (cvd - cvd.shift(window)) / cvd_shift.abs()

    price_sign = np.sign(price_roc)
    cvd_sign = np.sign(cvd_roc)
    # 부호 불일치 → divergence
    result: pd.Series = price_sign - cvd_sign  # type: ignore[assignment]
    # 정규화: -2→-1, 0→0, +2→+1
    result = result.clip(-1, 1)
    return result


def liquidation_cascade_score(
    liq_long: pd.Series,
    liq_short: pd.Series,
    open_interest: pd.Series,
) -> pd.Series:
    """Liquidation Cascade Score — (liq_long + liq_short) / OI.

    높은 값: 대량 청산 발생 (cascade 리스크).
    OI 대비 청산량이 비정상적으로 클 때 시장 스트레스 신호.

    Args:
        liq_long: 롱 청산량 시리즈 (USD).
        liq_short: 숏 청산량 시리즈 (USD).
        open_interest: 미결제약정 시리즈 (USD).

    Returns:
        Cascade score 시리즈 (0~).
    """
    total_liq: pd.Series = liq_long + liq_short  # type: ignore[assignment]
    oi_safe = open_interest.replace(0, np.nan)
    result: pd.Series = total_liq / oi_safe  # type: ignore[assignment]
    return result


def liquidation_asymmetry(
    liq_long: pd.Series,
    liq_short: pd.Series,
) -> pd.Series:
    """Liquidation Asymmetry — (liq_long - liq_short) / (liq_long + liq_short).

    양수: 롱 청산 우세 (시장 하락 압력).
    음수: 숏 청산 우세 (시장 상승 압력).

    Args:
        liq_long: 롱 청산량 시리즈 (USD).
        liq_short: 숏 청산량 시리즈 (USD).

    Returns:
        비대칭도 시리즈 (-1 ~ +1).
    """
    total: pd.Series = liq_long + liq_short  # type: ignore[assignment]
    total_safe = total.replace(0, np.nan)
    result: pd.Series = (liq_long - liq_short) / total_safe  # type: ignore[assignment]
    return result


def taker_cvd(
    taker_buy_base_volume: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Taker 기반 Cumulative Volume Delta.

    CVD = cumsum(taker_buy - taker_sell) = cumsum(2*taker_buy - volume).
    상승: 매수 주도, 하락: 매도 주도.

    Args:
        taker_buy_base_volume: Taker 매수 base asset 거래량 (Binance klines col 9).
        volume: 전체 거래량.

    Returns:
        누적 CVD 시리즈.
    """
    delta: pd.Series = 2.0 * taker_buy_base_volume - volume  # type: ignore[assignment]
    result: pd.Series = delta.cumsum()  # type: ignore[assignment]
    return result


def taker_buy_ratio(
    taker_buy_base_volume: pd.Series,
    volume: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Rolling taker buy ratio = rolling_sum(taker_buy) / rolling_sum(volume).

    > 0.5: buyer-dominant, < 0.5: seller-dominant.

    Args:
        taker_buy_base_volume: Taker 매수 base asset 거래량.
        volume: 전체 거래량.
        window: Rolling 윈도우 (기본 14).

    Returns:
        Taker buy ratio 시리즈 (0~1).
    """
    rolling_buy: pd.Series = taker_buy_base_volume.rolling(  # type: ignore[assignment]
        window=window, min_periods=window
    ).sum()
    rolling_vol: pd.Series = volume.rolling(  # type: ignore[assignment]
        window=window, min_periods=window
    ).sum()
    rolling_vol_safe = rolling_vol.replace(0, np.nan)
    result: pd.Series = rolling_buy / rolling_vol_safe  # type: ignore[assignment]
    return result


def order_flow_imbalance(
    taker_buy_volume: pd.Series,
    taker_sell_volume: pd.Series,
    window: int = 30,
) -> pd.Series:
    """Order Flow Imbalance Z-Score — Taker 매수/매도 불균형의 z-score.

    양수: 비정상적 매수 주도 (bullish).
    음수: 비정상적 매도 주도 (bearish).

    Args:
        taker_buy_volume: Taker 매수 거래량 시리즈.
        taker_sell_volume: Taker 매도 거래량 시리즈.
        window: Z-score rolling 윈도우 (기본 30일).

    Returns:
        Order flow imbalance z-score 시리즈.
    """
    imbalance: pd.Series = taker_buy_volume - taker_sell_volume  # type: ignore[assignment]
    rolling_mean = imbalance.rolling(window=window, min_periods=window).mean()
    rolling_std = imbalance.rolling(window=window, min_periods=window).std()
    rolling_std_safe = rolling_std.replace(0, np.nan)
    result: pd.Series = (imbalance - rolling_mean) / rolling_std_safe  # type: ignore[assignment]
    return result
