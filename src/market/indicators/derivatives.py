"""Crypto derivatives indicators (funding rate, open interest, basis, etc.).

크립토 파생상품 데이터(펀딩비, 미결제약정, 롱숏비 등)를 활용하는 지표 함수.
모든 함수는 stateless, vectorized이며 ``pd.Series`` in → ``pd.Series`` out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def funding_rate_ma(
    funding_rate: pd.Series,
    window: int,
) -> pd.Series:
    """펀딩비 이동평균.

    Args:
        funding_rate: 펀딩비 시리즈.
        window: Rolling 윈도우 크기.

    Returns:
        펀딩비 이동평균 시리즈.
    """
    result: pd.Series = funding_rate.rolling(  # type: ignore[assignment]
        window=window, min_periods=window
    ).mean()
    return result


def funding_zscore(
    funding_rate: pd.Series,
    ma_window: int,
    zscore_window: int,
) -> pd.Series:
    """펀딩비 z-score (이동평균 후 정규화).

    펀딩비의 이동평균을 계산한 뒤 rolling z-score로 정규화합니다.

    Args:
        funding_rate: 펀딩비 시리즈.
        ma_window: 이동평균 윈도우.
        zscore_window: z-score 정규화 윈도우.

    Returns:
        z-score 정규화된 펀딩비 시리즈.
    """
    avg_fr = funding_rate_ma(funding_rate, ma_window)
    rolling_mean = avg_fr.rolling(window=zscore_window, min_periods=zscore_window).mean()
    rolling_std = avg_fr.rolling(window=zscore_window, min_periods=zscore_window).std()
    rolling_std_safe = rolling_std.replace(0, np.nan)
    return (avg_fr - rolling_mean) / rolling_std_safe


def oi_momentum(
    open_interest: pd.Series,
    period: int,
) -> pd.Series:
    """미결제약정(OI) 변화율 (Rate of Change).

    Args:
        open_interest: 미결제약정 시리즈.
        period: ROC 계산 기간.

    Returns:
        OI 변화율 시리즈.
    """
    result: pd.Series = open_interest.pct_change(period)  # type: ignore[assignment]
    return result


def oi_price_divergence(
    close: pd.Series,
    open_interest: pd.Series,
    window: int,
) -> pd.Series:
    """OI-가격 다이버전스 감지.

    가격과 OI의 rolling 상관계수를 계산합니다.
    음의 상관: 다이버전스 (가격↑ OI↓ 또는 가격↓ OI↑).

    Args:
        close: 종가 시리즈.
        open_interest: 미결제약정 시리즈.
        window: Rolling 상관계수 윈도우.

    Returns:
        rolling 상관계수 시리즈 (-1 ~ +1).
    """
    result: pd.Series = close.rolling(window).corr(open_interest)  # type: ignore[assignment]
    return result


def basis_spread(
    spot_close: pd.Series,
    futures_close: pd.Series,
) -> pd.Series:
    """현물-선물 베이시스 스프레드 (%).

    (futures - spot) / spot * 100 으로 계산합니다.
    양수: 콘탱고 (선물 프리미엄), 음수: 백워데이션.

    Args:
        spot_close: 현물 종가 시리즈.
        futures_close: 선물 종가 시리즈.

    Returns:
        베이시스 스프레드 (%) 시리즈.
    """
    spot_safe = spot_close.replace(0, np.nan)
    result: pd.Series = (futures_close - spot_close) / spot_safe * 100  # type: ignore[assignment]
    return result


def ls_ratio_zscore(
    ls_ratio: pd.Series,
    window: int,
) -> pd.Series:
    """롱숏비 z-score.

    롱숏비를 rolling z-score로 정규화하여 극단적 포지셔닝을 감지합니다.

    Args:
        ls_ratio: 롱숏비 시리즈.
        window: z-score 정규화 윈도우.

    Returns:
        z-score 정규화된 롱숏비 시리즈.
    """
    rolling_mean = ls_ratio.rolling(window=window, min_periods=window).mean()
    rolling_std = ls_ratio.rolling(window=window, min_periods=window).std()
    rolling_std_safe = rolling_std.replace(0, np.nan)
    return (ls_ratio - rolling_mean) / rolling_std_safe


def liquidation_intensity(
    liq_volume: pd.Series,
    volume: pd.Series,
    window: int,
) -> pd.Series:
    """청산 강도 (Liquidation Intensity).

    청산 거래량의 총 거래량 대비 비율을 rolling 평균으로 계산합니다.

    Args:
        liq_volume: 청산 거래량 시리즈.
        volume: 총 거래량 시리즈.
        window: Rolling 윈도우.

    Returns:
        청산 강도 시리즈 (0 ~ 1).
    """
    volume_safe = volume.replace(0, np.nan)
    ratio: pd.Series = liq_volume / volume_safe  # type: ignore[assignment]
    result: pd.Series = ratio.rolling(  # type: ignore[assignment]
        window=window, min_periods=window
    ).mean()
    return result
