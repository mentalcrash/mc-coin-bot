"""Asymmetric Semivariance MR 전처리 모듈.

OHLCV 데이터에서 방향별 semivariance 및 비율 Z-score를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.market.indicators import (
    atr,
    drawdown,
    log_returns,
    realized_volatility,
    simple_returns,
    volatility_scalar,
)

if TYPE_CHECKING:
    from src.strategy.asym_semivar_mr.config import AsymSemivarMRConfig

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def _calculate_semivariance(
    returns: pd.Series,
    window: int,
    *,
    direction: str = "down",
) -> pd.Series:
    """Rolling semivariance 계산.

    Semivariance = E[min(r, 0)^2] (downside) 또는 E[max(r, 0)^2] (upside).

    Args:
        returns: 수익률 시리즈
        window: Rolling 윈도우 크기
        direction: "down" 또는 "up"

    Returns:
        Semivariance 시리즈
    """
    semi_returns = np.minimum(returns, 0.0) if direction == "down" else np.maximum(returns, 0.0)

    squared: pd.Series = semi_returns**2  # type: ignore[assignment]
    semivar: pd.Series = squared.rolling(  # type: ignore[assignment]
        window=window, min_periods=window
    ).mean()
    return semivar


def _calculate_semivar_ratio(
    down_semivar: pd.Series,
    up_semivar: pd.Series,
) -> pd.Series:
    """Semivariance ratio = downside / (downside + upside).

    ratio > 0.5 -> downside dominant (fear)
    ratio < 0.5 -> upside dominant (greed)

    Args:
        down_semivar: Downside semivariance
        up_semivar: Upside semivariance

    Returns:
        Semivariance ratio (0~1 범위, total=0이면 NaN)
    """
    total = down_semivar + up_semivar
    total_safe: pd.Series = total.replace(0, np.nan)  # type: ignore[assignment]
    return down_semivar / total_safe


def _calculate_rolling_zscore(
    series: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling Z-score 계산.

    Args:
        series: 입력 시리즈
        window: Rolling 윈도우 크기

    Returns:
        Z-score 시리즈
    """
    rolling_mean: pd.Series = series.rolling(  # type: ignore[assignment]
        window=window, min_periods=window
    ).mean()
    rolling_std: pd.Series = series.rolling(  # type: ignore[assignment]
        window=window, min_periods=window
    ).std(ddof=1)
    rolling_std_safe: pd.Series = rolling_std.replace(0, np.nan)  # type: ignore[assignment]
    return (series - rolling_mean) / rolling_std_safe


def preprocess(df: pd.DataFrame, config: AsymSemivarMRConfig) -> pd.DataFrame:
    """Asymmetric Semivariance MR feature 계산.

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - down_semivar: Downside semivariance
        - up_semivar: Upside semivariance
        - semivar_ratio: Downside / Total semivariance (0~1)
        - semivar_zscore: Semivar ratio의 rolling Z-score
        - realized_vol: 실현 변동성
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range
        - drawdown: 롤링 최고점 대비 드로다운

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: 전략 설정

    Returns:
        feature가 추가된 새 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
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
    returns = log_returns(close_series) if config.use_log_returns else simple_returns(close_series)
    result["returns"] = returns

    # 2. Downside semivariance
    down_sv = _calculate_semivariance(returns, window=config.semivar_window, direction="down")
    result["down_semivar"] = down_sv

    # 3. Upside semivariance
    up_sv = _calculate_semivariance(returns, window=config.semivar_window, direction="up")
    result["up_semivar"] = up_sv

    # 4. Semivariance ratio (downside / total)
    semivar_ratio = _calculate_semivar_ratio(down_sv, up_sv)
    result["semivar_ratio"] = semivar_ratio

    # 5. Semivar ratio Z-score (rolling standardization)
    result["semivar_zscore"] = _calculate_rolling_zscore(semivar_ratio, window=config.zscore_window)

    # 6. 실현 변동성
    realized_vol = realized_volatility(
        returns,
        window=config.mom_lookback,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = realized_vol

    # 7. 변동성 스케일러
    result["vol_scalar"] = volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 8. ATR 계산
    result["atr"] = atr(high_series, low_series, close_series, period=config.atr_period)

    # 9. 드로다운 계산
    result["drawdown"] = drawdown(close_series)

    # 디버그 로깅
    valid_data = result.dropna()
    if len(valid_data) > 0:
        ratio_mean = valid_data["semivar_ratio"].mean()
        z_mean = valid_data["semivar_zscore"].mean()
        vs_mean = valid_data["vol_scalar"].mean()
        logger.info(
            "AsymSemivarMR Indicators | Avg Ratio: %.4f, Avg Z-score: %.4f, Avg Vol Scalar: %.4f",
            ratio_mean,
            z_mean,
            vs_mean,
        )

    return result
