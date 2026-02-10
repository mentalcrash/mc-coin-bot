"""Variance Ratio Regime Preprocessor (Indicator Calculation).

Lo-MacKinlay Variance Ratio와 z-stat을 rolling window로 계산합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
"""

import logging

import numpy as np
import pandas as pd

from src.strategy.vol_regime.preprocessor import (
    calculate_atr,
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)
from src.strategy.vr_regime.config import VRRegimeConfig

logger = logging.getLogger(__name__)


def _rolling_variance_ratio(
    returns: pd.Series,
    window: int,
    k: int,
    use_heteroscedastic: bool,
) -> tuple[pd.Series, pd.Series]:
    """Rolling Variance Ratio 및 z-stat 계산.

    Lo-MacKinlay (1988) 방법론:
        VR(k) = Var(k-period returns) / (k * Var(1-period returns))

    Args:
        returns: 1-period 수익률 시리즈
        window: Rolling 윈도우 크기
        k: VR 집계 기간
        use_heteroscedastic: Lo-MacKinlay robust z-stat 사용 여부

    Returns:
        (vr_series, z_stat_series) 튜플
    """
    # k-period returns
    k_returns = returns.rolling(window=k, min_periods=k).sum()

    # Rolling variance of 1-period returns
    var_1: pd.Series = returns.rolling(window=window, min_periods=window).var(ddof=1)  # type: ignore[assignment]

    # Rolling variance of k-period returns
    var_k: pd.Series = k_returns.rolling(window=window, min_periods=window).var(ddof=1)  # type: ignore[assignment]

    # VR(k) = var_k / (k * var_1)
    var_1_safe: pd.Series = var_1.replace(0, np.nan)  # type: ignore[assignment]
    vr = var_k / (k * var_1_safe)

    # z-stat 계산
    if use_heteroscedastic:
        # Lo-MacKinlay heteroscedastic-robust z-stat
        # delta_j = sum((r_t - mu)^2 * (r_{t-j} - mu)^2) / (sum((r_t - mu)^2))^2
        # 간소화 근사: theta = 2(2k-1)(k-1) / (3k*n) for homoscedastic case
        # heteroscedastic: rolling 기반 근사
        n = window
        # Asymptotic variance under heteroscedasticity (simplified)
        theta = 2.0 * (2.0 * k - 1.0) * (k - 1.0) / (3.0 * k * n)
        theta_safe = np.maximum(theta, 1e-10)
        z_stat: pd.Series = (vr - 1.0) / np.sqrt(theta_safe)  # type: ignore[assignment]
    else:
        # Simple z-stat (IID assumption)
        n = window
        se = np.sqrt(2.0 * (k - 1.0) / n)
        se_safe = max(se, 1e-10)
        z_stat = (vr - 1.0) / se_safe  # type: ignore[assignment]

    return vr, z_stat


def preprocess(
    df: pd.DataFrame,
    config: VRRegimeConfig,
) -> pd.DataFrame:
    """Variance Ratio Regime 전처리 (지표 계산).

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - vr: Variance Ratio VR(k)
        - vr_z_stat: VR z-statistic
        - mom_direction: 모멘텀 방향 (sign of rolling sum)
        - realized_vol: 실현 변동성
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range
        - drawdown: 롤링 최고점 대비 드로다운

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: VR Regime 설정

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

    # 2. Variance Ratio 및 z-stat
    vr, z_stat = _rolling_variance_ratio(
        returns_series,
        window=config.vr_window,
        k=config.vr_k,
        use_heteroscedastic=config.use_heteroscedastic,
    )
    result["vr"] = vr
    result["vr_z_stat"] = z_stat

    # 3. 모멘텀 방향
    mom_sum = returns_series.rolling(
        window=config.mom_lookback,
        min_periods=config.mom_lookback,
    ).sum()
    result["mom_direction"] = np.sign(mom_sum)

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
        vr_mean = valid_data["vr"].mean()
        z_mean = valid_data["vr_z_stat"].mean()
        vs_mean = valid_data["vol_scalar"].mean()
        logger.info(
            "VR-Regime Indicators | Avg VR: %.4f, Avg z-stat: %.4f, Avg Vol Scalar: %.4f",
            vr_mean,
            z_mean,
            vs_mean,
        )

    return result
