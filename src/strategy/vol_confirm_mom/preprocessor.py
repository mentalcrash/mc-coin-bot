"""Volume-Confirmed Momentum preprocessor.

Computes momentum return, volume SMA crossover, and vol-target features.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.market.indicators import (
    atr,
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.vol_confirm_mom.config import VolConfirmMomConfig

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: VolConfirmMomConfig) -> pd.DataFrame:
    """Volume-Confirmed Momentum feature computation.

    Calculated Columns:
        - returns: Log returns
        - realized_vol: Annualized realized volatility
        - vol_scalar: Vol-target scalar
        - mom_return: Rolling sum of returns (momentum direction)
        - vol_sma_short: Short-term volume SMA
        - vol_sma_long: Long-term volume SMA
        - vol_rising: Boolean - short vol SMA > long vol SMA
        - vol_ratio: Short vol SMA / long vol SMA (conviction)
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

    # --- Momentum Return ---
    df["mom_return"] = returns.rolling(
        window=config.mom_lookback,
        min_periods=config.mom_lookback,
    ).sum()

    # --- Volume SMA Crossover ---
    vol_sma_short = volume.rolling(
        window=config.vol_short_window,
        min_periods=config.vol_short_window,
    ).mean()
    vol_sma_long = volume.rolling(
        window=config.vol_long_window,
        min_periods=config.vol_long_window,
    ).mean()

    df["vol_sma_short"] = vol_sma_short
    df["vol_sma_long"] = vol_sma_long
    df["vol_rising"] = vol_sma_short > vol_sma_long

    # --- Volume Ratio (conviction: how much stronger is short vol vs long) ---
    vol_ratio = vol_sma_short / vol_sma_long.clip(lower=1e-10)  # type: ignore[reportCallIssue]
    df["vol_ratio"] = vol_ratio.clip(upper=config.vol_ratio_clip)  # type: ignore[reportCallIssue]

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    # --- Drawdown (HEDGE_ONLY) ---
    df["drawdown"] = drawdown(close)

    return df
