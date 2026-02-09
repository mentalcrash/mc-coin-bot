"""HAR Volatility Overlay Preprocessor (Indicator Calculation).

이 모듈은 HAR-RV 전략에 필요한 지표를 계산합니다.
Parkinson volatility, HAR features, rolling OLS forecast, vol surprise를 계산합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization where possible
    - #12 Data Engineering: Log returns for internal calculation
    - #26 VectorBT Standards: Compatible output format
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.tsmom.preprocessor import (
    calculate_atr,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)

if TYPE_CHECKING:
    from src.strategy.har_vol.config import HARVolConfig

logger = logging.getLogger(__name__)


def calculate_parkinson_vol(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson volatility 계산 (range-based estimator).

    High-Low 범위를 사용한 변동성 추정기입니다.
    Close-to-Close 변동성보다 효율적인 추정이 가능합니다.

    Formula:
        sqrt(1 / (4 * ln(2)) * ln(high / low)^2)

    Args:
        high: 고가 시리즈
        low: 저가 시리즈

    Returns:
        Parkinson volatility 시리즈
    """
    log_hl_ratio = np.log(high / low)
    return pd.Series(
        np.sqrt(1.0 / (4.0 * np.log(2.0)) * log_hl_ratio**2),
        index=high.index,
        name="parkinson_vol",
    )


def calculate_har_features(
    parkinson_vol: pd.Series,
    daily_w: int,
    weekly_w: int,
    monthly_w: int,
) -> pd.DataFrame:
    """HAR features 계산 (rv_daily, rv_weekly, rv_monthly).

    HAR-RV 모델의 입력 feature를 생성합니다.
    각각 daily, weekly, monthly 윈도우의 rolling mean입니다.

    Args:
        parkinson_vol: Parkinson volatility 시리즈
        daily_w: Daily 윈도우 크기
        weekly_w: Weekly 윈도우 크기
        monthly_w: Monthly 윈도우 크기

    Returns:
        DataFrame with columns: rv_daily, rv_weekly, rv_monthly
    """
    rv_daily = parkinson_vol.rolling(window=daily_w, min_periods=daily_w).mean()
    rv_weekly = parkinson_vol.rolling(window=weekly_w, min_periods=weekly_w).mean()
    rv_monthly = parkinson_vol.rolling(window=monthly_w, min_periods=monthly_w).mean()

    return pd.DataFrame(
        {
            "rv_daily": rv_daily,
            "rv_weekly": rv_weekly,
            "rv_monthly": rv_monthly,
        },
        index=parkinson_vol.index,
    )


def calculate_har_forecast(
    parkinson_vol: pd.Series,
    features_df: pd.DataFrame,
    training_window: int,
) -> pd.Series:
    """HAR-RV rolling OLS forecast 계산.

    각 시점 t에서 [t - training_window : t] 구간의 데이터로 OLS를 학습하고,
    시점 t의 features로 t+1 시점의 변동성을 예측합니다.

    # NOTE: Rolling OLS requires loop — cannot vectorize arbitrary regression windows.
    # numpy.linalg.lstsq를 사용하여 각 윈도우에서 OLS를 수행합니다.
    # Pre-allocated array로 효율성을 확보합니다.

    Args:
        parkinson_vol: Parkinson volatility 시리즈 (target)
        features_df: HAR features DataFrame (rv_daily, rv_weekly, rv_monthly)
        training_window: OLS 학습 윈도우 크기

    Returns:
        HAR forecast 시리즈 (예측 변동성)
    """
    n = len(parkinson_vol)
    forecast = np.full(n, np.nan)

    # Pre-allocate feature matrix columns
    rv_daily_arr = features_df["rv_daily"].to_numpy()
    rv_weekly_arr = features_df["rv_weekly"].to_numpy()
    rv_monthly_arr = features_df["rv_monthly"].to_numpy()
    target_arr = parkinson_vol.to_numpy()

    # NOTE: Rolling OLS requires loop — cannot vectorize arbitrary regression windows.
    # Each iteration fits OLS on [t - training_window : t], then predicts at t.
    for t in range(training_window, n):
        start = t - training_window
        end = t

        # Training data: features at [start:end-1] predict target at [start+1:end]
        y_train = target_arr[start + 1 : end]

        # Feature matrix with intercept: [1, rv_daily, rv_weekly, rv_monthly]
        x_daily = rv_daily_arr[start : end - 1]
        x_weekly = rv_weekly_arr[start : end - 1]
        x_monthly = rv_monthly_arr[start : end - 1]

        # Skip if any NaN in training data
        if (
            np.any(np.isnan(y_train))
            or np.any(np.isnan(x_daily))
            or np.any(np.isnan(x_weekly))
            or np.any(np.isnan(x_monthly))
        ):
            continue

        x_train = np.column_stack([np.ones(len(y_train)), x_daily, x_weekly, x_monthly])

        # OLS: minimize ||y - X @ beta||^2
        result = np.linalg.lstsq(x_train, y_train, rcond=None)
        beta = result[0]

        # Predict at time t using features at t
        if not (
            np.isnan(rv_daily_arr[t]) or np.isnan(rv_weekly_arr[t]) or np.isnan(rv_monthly_arr[t])
        ):
            x_pred = np.array([1.0, rv_daily_arr[t], rv_weekly_arr[t], rv_monthly_arr[t]])
            forecast[t] = float(x_pred @ beta)

    return pd.Series(forecast, index=parkinson_vol.index, name="har_forecast")


def calculate_vol_surprise(
    realized: pd.Series,
    forecast: pd.Series,
) -> pd.Series:
    """Vol surprise 계산 (realized - forecast).

    양수: 실현 변동성이 예측보다 높음 (vol expansion → momentum)
    음수: 실현 변동성이 예측보다 낮음 (vol contraction → mean-reversion)

    Args:
        realized: 실현 변동성 시리즈
        forecast: 예측 변동성 시리즈

    Returns:
        Vol surprise 시리즈
    """
    return pd.Series(
        realized - forecast,
        index=realized.index,
        name="vol_surprise",
    )


def preprocess(
    df: pd.DataFrame,
    config: HARVolConfig,
) -> pd.DataFrame:
    """HAR Volatility Overlay 전처리 (지표 계산).

    OHLCV DataFrame에 HAR-RV 전략에 필요한 기술적 지표를 계산하여 추가합니다.

    Calculated Columns:
        - returns: 수익률 (로그)
        - parkinson_vol: Parkinson volatility
        - rv_daily: Daily rolling mean of Parkinson vol
        - rv_weekly: Weekly rolling mean of Parkinson vol
        - rv_monthly: Monthly rolling mean of Parkinson vol
        - har_forecast: HAR-RV OLS forecast
        - vol_surprise: realized - forecast
        - realized_vol: 연환산 실현 변동성
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
            필수 컬럼: close, high, low
        config: HAR Volatility 설정

    Returns:
        지표가 추가된 새로운 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    required_cols = {"close", "high", "low"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    # 원본 보존 (복사본 생성)
    result = df.copy()

    # OHLCV 컬럼을 float64로 변환 (Decimal 타입 처리)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 컬럼 추출
    close_series: pd.Series = result["close"]  # type: ignore[assignment]
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]

    # 1. 수익률 계산 (log returns)
    result["returns"] = calculate_returns(close_series, use_log=True)
    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. Parkinson volatility
    parkinson_vol = calculate_parkinson_vol(high_series, low_series)
    result["parkinson_vol"] = parkinson_vol

    # 3. HAR features (rv_daily, rv_weekly, rv_monthly)
    har_features = calculate_har_features(
        parkinson_vol,
        daily_w=config.daily_window,
        weekly_w=config.weekly_window,
        monthly_w=config.monthly_window,
    )
    result["rv_daily"] = har_features["rv_daily"]
    result["rv_weekly"] = har_features["rv_weekly"]
    result["rv_monthly"] = har_features["rv_monthly"]

    # 4. HAR forecast (rolling OLS)
    har_forecast = calculate_har_forecast(
        parkinson_vol,
        har_features,
        training_window=config.training_window,
    )
    result["har_forecast"] = har_forecast

    # 5. Vol surprise (realized - forecast)
    result["vol_surprise"] = calculate_vol_surprise(parkinson_vol, har_forecast)

    # 6. 실현 변동성 (연환산, weekly_window 기반)
    realized_vol = calculate_realized_volatility(
        returns_series,
        window=config.weekly_window,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = realized_vol

    # 7. 변동성 스케일러
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 8. ATR 계산 (Trailing Stop용)
    result["atr"] = calculate_atr(high_series, low_series, close_series)

    # 디버그: 지표 통계
    valid_data = result.dropna(subset=["vol_surprise"])
    if len(valid_data) > 0:
        vs_mean = valid_data["vol_surprise"].mean()
        vs_std = valid_data["vol_surprise"].std()
        forecast_mean = valid_data["har_forecast"].mean()
        logger.info(
            "HAR-Vol Indicators | Vol Surprise: mean=%.6f, std=%.6f, Forecast mean=%.6f",
            vs_mean,
            vs_std,
            forecast_mean,
        )

    return result
