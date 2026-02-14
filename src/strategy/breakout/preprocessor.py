"""Adaptive Breakout Preprocessor.

이 모듈은 Adaptive Breakout 전략에 필요한 지표를 계산합니다.
모든 계산은 벡터화된 연산을 사용합니다 (for 루프 금지).

Rules Applied:
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Broadcasting compatible
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.market.indicators import (
    atr,
    donchian_channel,
    log_returns,
    realized_volatility,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.breakout.config import AdaptiveBreakoutConfig


def calculate_volatility_ratio(
    current_vol: pd.Series,
    avg_vol: pd.Series,
    min_ratio: float = 0.5,
    max_ratio: float = 2.0,
) -> pd.Series:
    """변동성 비율을 계산합니다.

    현재 변동성 / 평균 변동성 비율로 포지션 크기를 조절합니다.

    Args:
        current_vol: 현재 변동성 Series
        avg_vol: 평균 변동성 Series
        min_ratio: 최소 비율
        max_ratio: 최대 비율

    Returns:
        클램핑된 변동성 비율 Series
    """
    ratio: pd.Series = current_vol / avg_vol.replace(0, np.nan)
    return ratio.clip(lower=min_ratio, upper=max_ratio)


def calculate_adaptive_threshold(
    atr_val: pd.Series,
    k_value: float,
    volatility_ratio: pd.Series | None = None,
) -> pd.Series:
    """적응형 임계값을 계산합니다.

    기본: ATR * k_value
    Adaptive: ATR * k_value / volatility_ratio (변동성 높을 때 임계값 낮춤)

    Args:
        atr_val: ATR Series
        k_value: ATR 배수
        volatility_ratio: 변동성 비율 (None이면 고정 임계값)

    Returns:
        적응형 임계값 Series
    """
    if volatility_ratio is not None:
        # 변동성이 높을 때 임계값을 낮춰 더 많은 진입 기회
        return atr_val * k_value / volatility_ratio
    return atr_val * k_value


def calculate_distance_to_band(
    close: pd.Series,
    upper_band: pd.Series,
    lower_band: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """가격과 밴드 간 거리(%)를 계산합니다.

    Args:
        close: 종가 Series
        upper_band: 상단 밴드
        lower_band: 하단 밴드

    Returns:
        (distance_to_upper, distance_to_lower) 튜플 (%)
    """
    distance_to_upper: pd.Series = (upper_band - close) / close * 100
    distance_to_lower: pd.Series = (close - lower_band) / close * 100

    return distance_to_upper, distance_to_lower


def preprocess(df: pd.DataFrame, config: AdaptiveBreakoutConfig) -> pd.DataFrame:
    """Adaptive Breakout 전략을 위한 데이터 전처리.

    원본 OHLCV 데이터에 필요한 모든 지표를 계산하여 추가합니다.

    Calculated Columns:
        - upper_band: Donchian Channel 상단
        - lower_band: Donchian Channel 하단
        - middle_band: Donchian Channel 중심선
        - atr: Average True Range
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러 (vol_target / realized_vol)
        - threshold: 돌파 확인 임계값 (ATR * k_value)
        - distance_to_upper: 상단 밴드까지 거리 (%)
        - distance_to_lower: 하단 밴드까지 거리 (%)

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: 전략 설정

    Returns:
        지표가 추가된 새로운 DataFrame
    """
    # 복사본 생성 (원본 보존)
    result = df.copy()

    # 컬럼 추출
    high_series: pd.Series = df["high"]  # type: ignore[assignment]
    low_series: pd.Series = df["low"]  # type: ignore[assignment]
    close_series: pd.Series = df["close"]  # type: ignore[assignment]

    # 1. Donchian Channel 계산
    upper_band, middle_band, lower_band = donchian_channel(
        high_series, low_series, config.channel_period
    )
    result["upper_band"] = upper_band
    result["lower_band"] = lower_band
    result["middle_band"] = middle_band

    # 2. ATR 계산
    atr_val = atr(high_series, low_series, close_series, config.atr_period)
    result["atr"] = atr_val

    # 3. 변동성 계산 (log returns → realized_volatility)
    returns_series = log_returns(close_series)
    rv = realized_volatility(
        returns_series,
        window=config.volatility_lookback,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = rv

    # 최소 변동성 클램프 적용
    clamped_vol = rv.clip(lower=config.min_volatility)

    # 변동성 스케일러 (vol_target / realized_vol)
    # CRITICAL: shift(1)로 전봉 변동성 사용 (현재 봉의 변동성은 실시간에서 알 수 없음)
    prev_clamped_vol = clamped_vol.shift(1)
    vol_scalar: pd.Series = config.vol_target / prev_clamped_vol
    result["vol_scalar"] = vol_scalar

    # 4. 적응형 임계값 계산
    if config.adaptive_threshold:
        # 평균 변동성 계산 (더 긴 윈도우)
        avg_vol: pd.Series = rv.rolling(
            window=config.volatility_lookback * 2,
            min_periods=config.volatility_lookback,
        ).mean()  # type: ignore[assignment]

        vol_ratio = calculate_volatility_ratio(rv, avg_vol)
        result["volatility_ratio"] = vol_ratio

        threshold = calculate_adaptive_threshold(atr_val, config.k_value, vol_ratio)
    else:
        result["volatility_ratio"] = 1.0
        threshold = calculate_adaptive_threshold(atr_val, config.k_value)

    result["threshold"] = threshold

    # 5. 밴드까지 거리 계산 (진단용)
    dist_upper, dist_lower = calculate_distance_to_band(close_series, upper_band, lower_band)
    result["distance_to_upper"] = dist_upper
    result["distance_to_lower"] = dist_lower

    # NOTE: Trailing Stop은 Portfolio 레이어(BacktestEngine)에서 처리됩니다.
    # ATR은 별도 지표로 이미 포함되어 있어 BacktestEngine에서 활용 가능.

    return result
