"""Hour Seasonality Preprocessor (Indicator Calculation).

Per-hour rolling t-stat과 volume confirm 지표를 계산합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
"""

import logging

import numpy as np
import pandas as pd

from src.strategy.hour_season.config import HourSeasonConfig
from src.strategy.vol_regime.preprocessor import (
    calculate_atr,
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)

logger = logging.getLogger(__name__)


def _compute_hour_t_stat(
    returns: pd.Series,  # type: ignore[type-arg]
    season_window_days: int,
) -> pd.Series:  # type: ignore[type-arg]
    """Per-hour rolling t-statistic 계산.

    1H bars에서 동일 시간대(24 bars 간격)의 과거 returns를 수집하고,
    t-stat = mean / (std / sqrt(n))을 계산합니다.

    Args:
        returns: 수익률 시리즈
        season_window_days: 윈도우 일수

    Returns:
        t-stat 시리즈
    """
    # 24 bars 간격으로 lagged same-hour returns 수집
    lags = [24 * k for k in range(1, season_window_days + 1)]
    lagged = pd.concat([returns.shift(lag) for lag in lags], axis=1)

    n_obs = lagged.count(axis=1)
    hour_mean = lagged.mean(axis=1)
    hour_std = lagged.std(axis=1, ddof=1)

    # t-stat: mean / (std / sqrt(n)), guard against division by zero
    safe_denom = pd.Series(hour_std / np.sqrt(n_obs)).replace(0, np.nan)
    t_stat: pd.Series = hour_mean / safe_denom  # type: ignore[assignment]

    return t_stat


def preprocess(
    df: pd.DataFrame,
    config: HourSeasonConfig,
) -> pd.DataFrame:
    """Hour Seasonality 전처리 (지표 계산).

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - hour_t_stat: Per-hour rolling t-statistic
        - rel_volume: Relative volume (volume / rolling median)
        - vol_confirm: Volume confirm (rel_volume >= threshold)
        - realized_vol: 실현 변동성
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range
        - drawdown: 롤링 최고점 대비 드로다운

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수, 1H freq)
        config: Hour Seasonality 설정

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
    volume_series: pd.Series = result["volume"]  # type: ignore[assignment]

    # 1. 수익률 계산
    result["returns"] = calculate_returns(
        close_series,
        use_log=config.use_log_returns,
    )
    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. Per-hour rolling t-stat
    result["hour_t_stat"] = _compute_hour_t_stat(
        returns_series,
        config.season_window_days,
    )

    # 3. Relative volume (volume / rolling median)
    vol_median = volume_series.rolling(
        window=config.vol_confirm_window,
        min_periods=config.vol_confirm_window,
    ).median()
    vol_median_safe = vol_median.replace(0, np.nan)
    result["rel_volume"] = volume_series / vol_median_safe

    # 4. Volume confirm flag
    rel_vol_series: pd.Series = result["rel_volume"]  # type: ignore[assignment]
    result["vol_confirm"] = rel_vol_series >= config.vol_confirm_threshold

    # 5. 실현 변동성
    realized_vol = calculate_realized_volatility(
        returns_series,
        window=max(24, config.atr_period),
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
    valid_data = result.dropna(subset=["hour_t_stat"])
    if len(valid_data) > 0:
        ts_mean = valid_data["hour_t_stat"].abs().mean()
        sig_pct = (valid_data["hour_t_stat"].abs() > config.t_stat_threshold).mean() * 100
        vs_mean = valid_data["vol_scalar"].mean()
        msg = "Hour-Season Indicators | Avg |t-stat|: %.2f, Significant%%: %.1f%%, Avg Vol Scalar: %.4f"
        logger.info(msg, ts_mean, sig_pct, vs_mean)

    return result
