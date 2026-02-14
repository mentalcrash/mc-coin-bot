"""Volume MACD Momentum preprocessor.

Computes Volume MACD, histogram, price momentum, and vol-target features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.market.indicators import (
    atr,
    drawdown,
    ema,
    log_returns,
    realized_volatility,
    volatility_scalar,
    volume_macd,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.vmacd_mom.config import VmacdMomConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: VmacdMomConfig) -> pd.DataFrame:
    """Volume MACD Momentum feature computation.

    Calculated Columns:
        - returns: Log returns
        - realized_vol: Annualized realized volatility
        - vol_scalar: Vol-target scalar
        - vmacd_line: Volume MACD line
        - vmacd_signal_line: Volume MACD signal line
        - vmacd_hist: Volume MACD histogram
        - mom_return: Rolling momentum return
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
    vol_series: pd.Series = df["volume"]  # type: ignore[assignment]

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

    # --- Volume MACD ---
    # volume_macd returns a single Series (fast EMA - slow EMA)
    vmacd_line = volume_macd(vol_series, fast=config.vmacd_fast, slow=config.vmacd_slow)
    df["vmacd_line"] = vmacd_line

    # Signal line: EMA of VMACD line
    vmacd_signal_line = ema(vmacd_line, span=config.vmacd_signal)
    df["vmacd_signal_line"] = vmacd_signal_line

    # Histogram: VMACD - Signal
    df["vmacd_hist"] = vmacd_line - vmacd_signal_line

    # --- Momentum Return ---
    df["mom_return"] = returns.rolling(
        window=config.mom_lookback,
        min_periods=config.mom_lookback,
    ).sum()

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    # --- Drawdown (HEDGE_ONLY) ---
    df["drawdown"] = drawdown(close)

    return df
