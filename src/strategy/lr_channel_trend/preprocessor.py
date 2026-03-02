"""LR-Channel Multi-Scale Trend 전처리 모듈.

Rolling OLS 선형회귀 채널 x 3스케일(20/60/150) feature를 계산한다.
각 스케일에서 regression_end +/- multiplier * residual_std로 상/하단 밴드를 구성.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.lr_channel_trend.config import LrChannelTrendConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def _lr_regression_end(vals: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> float:
    """Rolling window 내 OLS 회귀선의 마지막 점 값을 반환한다.

    Args:
        vals: rolling window 내 close 가격 배열 (raw=True).

    Returns:
        회귀선의 endpoint 값. ss_xx == 0이면 NaN.
    """
    n = len(vals)
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    y_mean = vals.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx == 0:
        return np.nan
    slope = np.sum((x - x_mean) * (vals - y_mean)) / ss_xx
    intercept = y_mean - slope * x_mean
    return intercept + slope * (n - 1)


def _lr_residual_std(vals: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> float:
    """Rolling window 내 OLS 잔차의 표준편차(RMSE)를 반환한다.

    Args:
        vals: rolling window 내 close 가격 배열 (raw=True).

    Returns:
        잔차 RMSE. ss_xx == 0이면 NaN.
    """
    n = len(vals)
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    y_mean = vals.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx == 0:
        return np.nan
    slope = np.sum((x - x_mean) * (vals - y_mean)) / ss_xx
    intercept = y_mean - slope * x_mean
    y_hat = intercept + slope * x
    return float(np.sqrt(np.mean((vals - y_hat) ** 2)))


def preprocess(df: pd.DataFrame, config: LrChannelTrendConfig) -> pd.DataFrame:
    """LR-Channel Multi-Scale Trend feature 계산.

    Calculated Columns:
        - lr_upper_{s}, lr_lower_{s}: 3-scale Linear Regression Channel
        - returns: log return
        - realized_vol: 연환산 실현 변동성
        - vol_scalar: 변동성 스케일러
        - drawdown: HEDGE_ONLY용 drawdown

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

    df = df.copy()

    close: pd.Series = df["close"]  # type: ignore[assignment]

    scales = (config.scale_short, config.scale_mid, config.scale_long)

    # --- 3-Scale Linear Regression Channels ---
    for s in scales:
        reg_end = close.rolling(s).apply(_lr_regression_end, raw=True)
        res_std = close.rolling(s).apply(_lr_residual_std, raw=True)
        df[f"lr_upper_{s}"] = reg_end + config.channel_multiplier * res_std
        df[f"lr_lower_{s}"] = reg_end - config.channel_multiplier * res_std

    # --- Returns ---
    returns = log_returns(close)
    df["returns"] = returns

    # --- Realized Volatility ---
    realized_vol = realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    df["realized_vol"] = realized_vol

    # --- Vol Scalar ---
    df["vol_scalar"] = volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
