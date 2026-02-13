"""Trend Quality Momentum preprocessor.

Computes linear regression R^2 and slope on rolling windows,
plus standard vol-target and drawdown features.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.vol_regime.preprocessor import (
    calculate_atr,
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)

if TYPE_CHECKING:
    from src.strategy.trend_quality_mom.config import TrendQualityMomConfig

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def _rolling_regression(
    close: pd.Series,
    lookback: int,
) -> tuple[pd.Series, pd.Series]:
    """Compute rolling OLS slope and R-squared on close prices.

    Uses vectorized rolling.apply with numpy polyfit for efficiency.
    R^2 measures the 'quality' of the linear trend in the lookback window.

    Args:
        close: Close price series.
        lookback: Rolling window size.

    Returns:
        Tuple of (slope_series, r2_series).
    """
    x = np.arange(lookback, dtype=float)
    x_mean = x.mean()
    ss_xx = float(np.sum((x - x_mean) ** 2))
    _ss_epsilon = 1e-15  # Minimum SS_yy to avoid near-zero division

    def _r2_apply(window_data: np.ndarray) -> float:  # type: ignore[type-arg]
        y = window_data.astype(float)
        y_mean = y.mean()
        ss_yy = float(np.sum((y - y_mean) ** 2))
        if ss_yy < _ss_epsilon:
            return 0.0
        ss_xy = float(np.sum((x - x_mean) * (y - y_mean)))
        r_squared = (ss_xy**2) / (ss_xx * ss_yy)
        return float(np.clip(r_squared, 0.0, 1.0))

    def _slope_apply(window_data: np.ndarray) -> float:  # type: ignore[type-arg]
        y = window_data.astype(float)
        y_mean = y.mean()
        ss_xy = float(np.sum((x - x_mean) * (y - y_mean)))
        return ss_xy / ss_xx if ss_xx > 0 else 0.0

    r2_series: pd.Series = close.rolling(  # type: ignore[assignment]
        window=lookback,
        min_periods=lookback,
    ).apply(_r2_apply, raw=True)

    slope_series: pd.Series = close.rolling(  # type: ignore[assignment]
        window=lookback,
        min_periods=lookback,
    ).apply(_slope_apply, raw=True)

    return (
        pd.Series(slope_series, index=close.index),
        pd.Series(r2_series, index=close.index),
    )


def preprocess(df: pd.DataFrame, config: TrendQualityMomConfig) -> pd.DataFrame:
    """Trend Quality Momentum feature computation.

    Calculated Columns:
        - returns: Log returns
        - realized_vol: Annualized realized volatility
        - vol_scalar: Vol-target scalar
        - reg_slope: Rolling OLS slope of close
        - r_squared: Rolling R^2 (trend quality, 0~1)
        - mom_return: Rolling sum of returns (momentum direction)
        - atr: Average True Range
        - drawdown: Peak drawdown (for HEDGE_ONLY)

    Args:
        df: OHLCV DataFrame
        config: Strategy configuration

    Returns:
        DataFrame with features added

    Raises:
        ValueError: If required columns are missing
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    df = df.copy()

    close: pd.Series = df["close"]  # type: ignore[assignment]
    high: pd.Series = df["high"]  # type: ignore[assignment]
    low: pd.Series = df["low"]  # type: ignore[assignment]

    # --- Returns ---
    returns = calculate_returns(close)
    df["returns"] = returns

    # --- Realized Volatility ---
    realized_vol = calculate_realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    df["realized_vol"] = realized_vol

    # --- Vol Scalar ---
    df["vol_scalar"] = calculate_volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # --- Linear Regression: Slope + R^2 ---
    slope, r2 = _rolling_regression(close, config.regression_lookback)
    df["reg_slope"] = slope
    df["r_squared"] = r2

    # --- Momentum Return (direction confirmation) ---
    df["mom_return"] = returns.rolling(
        window=config.mom_lookback,
        min_periods=config.mom_lookback,
    ).sum()

    # --- ATR ---
    df["atr"] = calculate_atr(high, low, close, period=config.atr_period)

    # --- Drawdown (HEDGE_ONLY) ---
    df["drawdown"] = calculate_drawdown(close)

    return df
