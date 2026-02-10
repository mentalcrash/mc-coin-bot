"""OU Mean Reversion Preprocessor (Indicator Calculation).

Rolling OLS로 OU process 파라미터(theta, half-life, mu)를 추정하고
Z-score를 계산합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
"""

import logging

import numpy as np
import pandas as pd

from src.strategy.ou_meanrev.config import OUMeanRevConfig
from src.strategy.vol_regime.preprocessor import (
    calculate_atr,
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)

logger = logging.getLogger(__name__)


def _rolling_ou_params(
    close: pd.Series,
    window: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Rolling OU parameter estimation via OLS.

    For each rolling window:
        y = delta_price = price[t] - price[t-1]
        x = price[t-1]
        OLS: y = a + b*x
        theta = -log(1 + b) (when b < 0, theta > 0 = mean reverting)
        half_life = ln(2) / theta
        mu = -a / b (long-run mean)

    Vectorized approach using rolling covariance and variance for OLS beta:
        b = Cov(x, y) / Var(x)
        a = mean(y) - b * mean(x)

    Args:
        close: 종가 시리즈
        window: Rolling 윈도우 크기

    Returns:
        (theta, half_life, mu) 시리즈 튜플
    """
    price_lag = close.shift(1)
    delta = close - price_lag

    # Rolling OLS coefficients: b = Cov(x, y) / Var(x), a = E[y] - b * E[x]
    xy_cov: pd.Series = delta.rolling(window, min_periods=window).cov(price_lag)  # type: ignore[assignment]
    x_var: pd.Series = price_lag.rolling(window, min_periods=window).var()  # type: ignore[assignment]

    # Avoid division by zero
    x_var_safe: pd.Series = x_var.replace(0, np.nan)  # type: ignore[assignment]
    b = xy_cov / x_var_safe

    a: pd.Series = (  # type: ignore[assignment]
        delta.rolling(window, min_periods=window).mean()
        - b * price_lag.rolling(window, min_periods=window).mean()
    )

    # OU parameters
    # theta = -log(1 + b) where b should be negative for mean reversion
    # Clip b to (-1 + eps, -eps) for valid log computation
    b_clipped: pd.Series = b.clip(upper=-1e-10)  # type: ignore[assignment]
    theta: pd.Series = -np.log(1.0 + b_clipped)  # type: ignore[assignment]
    theta = theta.clip(lower=1e-10)  # type: ignore[assignment]

    half_life: pd.Series = np.log(2.0) / theta  # type: ignore[assignment]

    # Long-run mean mu = -a / b
    b_safe: pd.Series = b.replace(0, np.nan)  # type: ignore[assignment]
    mu: pd.Series = -a / b_safe  # type: ignore[assignment]

    return theta, half_life, mu


def preprocess(
    df: pd.DataFrame,
    config: OUMeanRevConfig,
) -> pd.DataFrame:
    """OU Mean Reversion 전처리 (지표 계산).

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - theta: OU mean reversion speed
        - half_life: Mean reversion half-life (bars)
        - ou_mu: OU long-run mean
        - ou_zscore: (close - ou_mu) / rolling_std(close, ou_window)
        - realized_vol: 실현 변동성
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range
        - drawdown: 롤링 최고점 대비 드로다운

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: OU Mean Reversion 설정

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

    # 2. OU process 파라미터 추정
    theta, half_life, ou_mu = _rolling_ou_params(
        close_series,
        window=config.ou_window,
    )
    result["theta"] = theta
    result["half_life"] = half_life
    result["ou_mu"] = ou_mu

    # 3. OU Z-score: (close - ou_mu) / rolling_std(close, ou_window)
    rolling_std: pd.Series = close_series.rolling(  # type: ignore[assignment]
        window=config.ou_window,
        min_periods=config.ou_window,
    ).std()
    rolling_std_safe: pd.Series = rolling_std.replace(0, np.nan)  # type: ignore[assignment]
    result["ou_zscore"] = (close_series - ou_mu) / rolling_std_safe

    # 4. 실현 변동성
    realized_vol = calculate_realized_volatility(
        returns_series,
        window=config.mom_lookback,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = realized_vol

    # 5. 변동성 스케일러
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 6. ATR 계산
    result["atr"] = calculate_atr(
        high_series,
        low_series,
        close_series,
        period=config.atr_period,
    )

    # 7. 드로다운 계산
    result["drawdown"] = calculate_drawdown(close_series)

    # 디버그: 지표 통계
    valid_data = result.dropna()
    if len(valid_data) > 0:
        hl_mean = valid_data["half_life"].mean()
        z_mean = valid_data["ou_zscore"].mean()
        vs_mean = valid_data["vol_scalar"].mean()
        logger.info(
            "OU-MeanRev Indicators | Avg Half-Life: %.2f, Avg Z-score: %.4f, Avg Vol Scalar: %.4f",
            hl_mean,
            z_mean,
            vs_mean,
        )

    return result
