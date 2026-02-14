"""VWAP Trend Crossover preprocessor.

Computes rolling VWAP (volume-weighted average price) on short and long windows,
plus standard vol-target and drawdown features.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from src.market.indicators import (
    atr,
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    from src.strategy.vwap_trend_cross.config import VwapTrendCrossConfig

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def _rolling_vwap(
    close: pd.Series,
    volume: pd.Series,
    window: int,
) -> pd.Series:
    """Compute rolling VWAP (Volume-Weighted Average Price).

    VWAP = sum(close * volume, window) / sum(volume, window)

    Args:
        close: Close price series.
        volume: Volume series.
        window: Rolling window size.

    Returns:
        Rolling VWAP series.
    """
    price_volume = close * volume
    sum_pv = price_volume.rolling(window=window, min_periods=window).sum()
    sum_v = volume.rolling(window=window, min_periods=window).sum()
    # Avoid division by zero
    vwap = sum_pv / sum_v.clip(lower=1e-10)  # type: ignore[reportCallIssue]
    return pd.Series(vwap, index=close.index)


def preprocess(df: pd.DataFrame, config: VwapTrendCrossConfig) -> pd.DataFrame:
    """VWAP Trend Crossover feature computation.

    Calculated Columns:
        - returns: Log returns
        - realized_vol: Annualized realized volatility
        - vol_scalar: Vol-target scalar
        - vwap_short: Short-term rolling VWAP
        - vwap_long: Long-term rolling VWAP
        - vwap_spread: Normalized spread (vwap_short - vwap_long) / close
        - atr: Average True Range
        - drawdown: Peak drawdown

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
    volume: pd.Series = df["volume"]  # type: ignore[assignment]

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

    # --- Rolling VWAP ---
    vwap_short = _rolling_vwap(close, volume, config.vwap_short_window)
    vwap_long = _rolling_vwap(close, volume, config.vwap_long_window)
    df["vwap_short"] = vwap_short
    df["vwap_long"] = vwap_long

    # --- VWAP Spread (normalized by close) ---
    # Positive spread = short VWAP above long VWAP = bullish
    spread = (vwap_short - vwap_long) / close.clip(lower=1e-10)
    df["vwap_spread"] = spread.clip(lower=-config.spread_clip, upper=config.spread_clip)

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    # --- Drawdown (HEDGE_ONLY) ---
    df["drawdown"] = drawdown(close)

    return df
