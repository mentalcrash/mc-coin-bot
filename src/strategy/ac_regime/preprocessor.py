"""Autocorrelation Regime-Adaptive Preprocessor (Indicator Calculation).

Rolling autocorrelation을 벡터화된 연산으로 계산합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
"""

import logging

import numpy as np
import pandas as pd

from src.strategy.ac_regime.config import ACRegimeConfig
from src.strategy.vol_regime.preprocessor import (
    calculate_atr,
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)

logger = logging.getLogger(__name__)


def calculate_rolling_autocorrelation(
    returns: pd.Series,
    window: int,
    lag: int,
) -> pd.Series:
    """벡터화된 rolling autocorrelation 계산.

    rolling().apply() 대신 수동 벡터화로 성능 최적화.

    Formula:
        rho = cov(x, x_lag) / sqrt(var(x) * var(x_lag))
        cov = mean((x - mean_x) * (x_lag - mean_x_lag))
        var_x = mean((x - mean_x)^2)
        var_x_lag = mean((x_lag - mean_x_lag)^2)

    Args:
        returns: 수익률 시리즈
        window: Rolling 윈도우 크기
        lag: Autocorrelation lag

    Returns:
        Rolling autocorrelation 시리즈
    """
    x = returns
    x_lag = returns.shift(lag)

    mean_x = x.rolling(window=window, min_periods=window).mean()
    mean_x_lag = x_lag.rolling(window=window, min_periods=window).mean()

    cov = ((x - mean_x) * (x_lag - mean_x_lag)).rolling(window=window, min_periods=window).mean()

    var_x = ((x - mean_x) ** 2).rolling(window=window, min_periods=window).mean()
    var_x_lag = ((x_lag - mean_x_lag) ** 2).rolling(window=window, min_periods=window).mean()

    denom = np.sqrt(var_x * var_x_lag)
    denom_safe = denom.replace(0, np.nan)
    rho: pd.Series = (cov / denom_safe).clip(-1.0, 1.0)  # type: ignore[assignment]
    return rho


def preprocess(
    df: pd.DataFrame,
    config: ACRegimeConfig,
) -> pd.DataFrame:
    """Autocorrelation Regime-Adaptive 전처리 (지표 계산).

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - ac_rho: Rolling autocorrelation
        - sig_bound: Bartlett significance bound (상수)
        - mom_direction: 모멘텀 방향 (sign of rolling sum)
        - realized_vol: 실현 변동성
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range
        - drawdown: 롤링 최고점 대비 드로다운

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: AC Regime 설정

    Returns:
        지표가 추가된 새로운 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    result = df.copy()

    # OHLCV 컬럼을 float64로 변환
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    close_series: pd.Series = result["close"]  # type: ignore[assignment]
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]

    # 1. 수익률 계산
    result["returns"] = calculate_returns(
        close_series,
        use_log=config.use_log_returns,
    )
    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. Rolling autocorrelation (벡터화)
    result["ac_rho"] = calculate_rolling_autocorrelation(
        returns_series,
        window=config.ac_window,
        lag=config.ac_lag,
    )

    # 3. Significance bound (Bartlett: z / sqrt(N))
    result["sig_bound"] = config.significance_z / np.sqrt(config.ac_window)

    # 4. 모멘텀 방향
    mom_sum = returns_series.rolling(
        window=config.mom_lookback,
        min_periods=config.mom_lookback,
    ).sum()
    result["mom_direction"] = np.sign(mom_sum)

    # 5. 실현 변동성
    realized_vol = calculate_realized_volatility(
        returns_series,
        window=config.mom_lookback,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = realized_vol

    # 6. 변동성 스케일러
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 7. ATR 계산
    result["atr"] = calculate_atr(
        high_series,
        low_series,
        close_series,
        period=config.atr_period,
    )

    # 8. 드로다운 계산
    result["drawdown"] = calculate_drawdown(close_series)

    # 디버그: 지표 통계
    valid_data = result.dropna()
    if len(valid_data) > 0:
        rho_mean = valid_data["ac_rho"].mean()
        vs_mean = valid_data["vol_scalar"].mean()
        logger.info(
            "AC-Regime Indicators | Avg AC Rho: %.4f, Sig Bound: %.4f, Avg Vol Scalar: %.4f",
            rho_mean,
            float(valid_data["sig_bound"].iloc[0]),
            vs_mean,
        )

    return result
