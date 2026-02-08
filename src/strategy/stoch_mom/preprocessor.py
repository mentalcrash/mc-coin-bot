"""Stochastic Momentum Hybrid Preprocessor (Indicator Calculation).

Calculates Stochastic %K/%D, SMA trend filter, ATR, realized volatility,
vol_scalar, and vol_ratio for dynamic position sizing.

Reused Functions:
    - calculate_returns, calculate_realized_volatility,
      calculate_volatility_scalar, calculate_atr: TSMOM module
    - calculate_drawdown: TSMOM module

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
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
    from src.strategy.stoch_mom.config import StochMomConfig

logger = logging.getLogger(__name__)


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int,
    d_period: int,
) -> tuple[pd.Series, pd.Series]:
    """Calculate Stochastic %K and %D.

    %K measures the current close relative to the high-low range
    over k_period. %D is the SMA of %K over d_period.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: Lookback period for %K
        d_period: Smoothing period for %D

    Returns:
        Tuple of (%K series, %D series), both in 0-100 range
    """
    lowest_low: pd.Series = low.rolling(window=k_period).min()  # type: ignore[assignment]
    highest_high: pd.Series = high.rolling(window=k_period).max()  # type: ignore[assignment]

    # %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
    range_hl = highest_high - lowest_low
    # Replace 0 range with NaN to avoid division by zero
    range_safe: pd.Series = range_hl.replace(0, np.nan)  # type: ignore[assignment]
    pct_k: pd.Series = 100 * (close - lowest_low) / range_safe  # type: ignore[assignment]

    # %D = SMA of %K over d_period
    pct_d: pd.Series = pct_k.rolling(window=d_period).mean()  # type: ignore[assignment]

    return (
        pd.Series(pct_k, index=close.index, name="pct_k"),
        pd.Series(pct_d, index=close.index, name="pct_d"),
    )


def calculate_sma(close: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average.

    Args:
        close: Close price series
        period: SMA period

    Returns:
        SMA series
    """
    sma: pd.Series = close.rolling(window=period).mean()  # type: ignore[assignment]
    return pd.Series(sma, index=close.index, name="sma")


def calculate_vol_ratio(
    atr: pd.Series,
    close: pd.Series,
    min_ratio: float,
    max_ratio: float,
) -> pd.Series:
    """Calculate ATR-based volatility ratio for dynamic sizing.

    vol_ratio = (atr / close), clipped to [min_ratio, max_ratio].
    Higher ATR relative to price -> larger position sizing factor.

    Args:
        atr: ATR series
        close: Close price series
        min_ratio: Minimum position size ratio
        max_ratio: Maximum position size ratio

    Returns:
        Vol ratio series, clipped to [min_ratio, max_ratio]
    """
    raw_ratio = atr / close
    clipped: pd.Series = raw_ratio.clip(lower=min_ratio, upper=max_ratio)  # type: ignore[assignment]
    return pd.Series(clipped, index=close.index, name="vol_ratio")


def preprocess(
    df: pd.DataFrame,
    config: StochMomConfig,
) -> pd.DataFrame:
    """Stochastic Momentum Hybrid preprocessing (indicator calculation).

    Calculates all technical indicators needed for signal generation.
    All computations are vectorized (no loops).

    Calculated Columns:
        - pct_k: Stochastic %K (0-100)
        - pct_d: Stochastic %D (0-100)
        - sma: Simple Moving Average
        - atr: Average True Range
        - returns: Log returns
        - realized_vol: Realized volatility (annualized)
        - vol_scalar: Volatility scalar (vol_target / realized_vol)
        - vol_ratio: ATR/close ratio, clipped to [min_vol_ratio, max_vol_ratio]

    Args:
        df: OHLCV DataFrame (DatetimeIndex required)
            Required columns: open, high, low, close
        config: Stochastic Momentum config

    Returns:
        New DataFrame with indicators added

    Raises:
        ValueError: When required columns are missing
    """
    # Input validation
    required_cols = {"open", "high", "low", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    # Preserve original (create copy)
    result = df.copy()

    # Convert OHLCV columns to float64 (handles Decimal types)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # Extract columns (explicit Series typing)
    close_series: pd.Series = result["close"]  # type: ignore[assignment]
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]

    # 1. Stochastic %K/%D
    pct_k, pct_d = calculate_stochastic(
        high_series,
        low_series,
        close_series,
        k_period=config.k_period,
        d_period=config.d_period,
    )
    result["pct_k"] = pct_k
    result["pct_d"] = pct_d

    # 2. SMA (trend filter)
    result["sma"] = calculate_sma(close_series, config.sma_period)

    # 3. ATR
    result["atr"] = calculate_atr(
        high_series,
        low_series,
        close_series,
        period=config.atr_period,
    )

    # 4. Log returns
    result["returns"] = calculate_returns(close_series, use_log=True)

    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 5. Realized volatility (annualized)
    vol_window = max(config.k_period, config.sma_period)
    result["realized_vol"] = calculate_realized_volatility(
        returns_series,
        window=vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 6. Volatility scalar (vol_target / realized_vol)
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 7. Vol ratio (ATR / close, clipped)
    atr_series: pd.Series = result["atr"]  # type: ignore[assignment]
    result["vol_ratio"] = calculate_vol_ratio(
        atr_series,
        close_series,
        min_ratio=config.min_vol_ratio,
        max_ratio=config.max_vol_ratio,
    )

    # Debug logging
    valid_data = result.dropna()
    if len(valid_data) > 0:
        k_min = valid_data["pct_k"].min()
        k_max = valid_data["pct_k"].max()
        vr_min = valid_data["vol_ratio"].min()
        vr_max = valid_data["vol_ratio"].max()
        logger.info(
            "Stoch-Mom Indicators | %%K: [%.1f, %.1f], Vol Ratio: [%.3f, %.3f]",
            k_min,
            k_max,
            vr_min,
            vr_max,
        )

    return result
