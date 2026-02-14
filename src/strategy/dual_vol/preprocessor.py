"""Dual Volatility Trend preprocessor.

Computes Yang-Zhang vol, Parkinson vol, their ratio, and momentum features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.market.indicators import (
    atr,
    drawdown,
    ema,
    log_returns,
    parkinson_volatility,
    realized_volatility,
    volatility_scalar,
    yang_zhang_volatility,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.dual_vol.config import DualVolConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: DualVolConfig) -> pd.DataFrame:
    """Dual Volatility Trend feature computation.

    Calculated Columns:
        - returns: Log returns
        - realized_vol: Annualized realized volatility
        - vol_scalar: Vol-target scalar
        - yz_vol: Yang-Zhang volatility
        - park_vol: Parkinson volatility (rolling)
        - vol_ratio: YZ / Parkinson ratio (smoothed)
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
    open_: pd.Series = df["open"]  # type: ignore[assignment]

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

    # --- Yang-Zhang Volatility ---
    yz_vol = yang_zhang_volatility(
        open_,
        high,
        low,
        close,
        window=config.vol_estimator_window,
    )
    df["yz_vol"] = yz_vol

    # --- Parkinson Volatility (rolling) ---
    park_raw = parkinson_volatility(high, low)
    park_vol = park_raw.rolling(
        window=config.vol_estimator_window,
        min_periods=config.vol_estimator_window,
    ).mean()
    df["park_vol"] = park_vol

    # --- Vol Ratio: YZ / Parkinson (smoothed) ---
    raw_ratio = yz_vol / np.clip(park_vol, 1e-10, None)
    # Smooth with EMA to reduce noise
    df["vol_ratio"] = ema(raw_ratio, span=config.ratio_smooth)

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
