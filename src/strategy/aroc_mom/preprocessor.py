"""Adaptive ROC Momentum preprocessor.

Computes adaptive lookback ROC, volatility rank, and vol-target features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.market.indicators import (
    atr,
    drawdown,
    log_returns,
    realized_volatility,
    roc,
    vol_percentile_rank,
    volatility_scalar,
)

if TYPE_CHECKING:
    from src.strategy.aroc_mom.config import ArocMomConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: ArocMomConfig) -> pd.DataFrame:
    """Adaptive ROC Momentum feature computation.

    Calculated Columns:
        - returns: Log returns
        - realized_vol: Annualized realized volatility
        - vol_scalar: Vol-target scalar
        - vol_rank: Volatility percentile rank (0~1)
        - adaptive_lookback: Dynamically computed ROC lookback
        - adaptive_roc: ROC computed with adaptive lookback
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

    # --- Volatility Percentile Rank (0~1) ---
    v_rank = vol_percentile_rank(realized_vol, window=config.vol_rank_window)
    df["vol_rank"] = v_rank

    # --- Adaptive Lookback: high vol -> fast, low vol -> slow ---
    # lookback = slow - (slow - fast) * vol_rank
    lookback_range = config.slow_lookback - config.fast_lookback
    # Multiply Series first to ensure Series result type
    adaptive_lb = v_rank.fillna(0.5) * (-lookback_range) + config.slow_lookback
    adaptive_lb_int = np.clip(adaptive_lb, config.fast_lookback, config.slow_lookback)
    df["adaptive_lookback"] = adaptive_lb_int

    # --- Adaptive ROC: compute ROC for each row using its adaptive lookback ---
    # Vectorized approach: compute ROC at each candidate lookback, then select
    roc_fast = roc(close, period=config.fast_lookback)
    roc_slow = roc(close, period=config.slow_lookback)
    roc_mid = roc(close, period=(config.fast_lookback + config.slow_lookback) // 2)

    # Interpolate ROC based on adaptive lookback fraction
    lb_frac = (adaptive_lb_int - config.fast_lookback) / max(lookback_range, 1)
    # Blend: fast weight when lb_frac is low, slow weight when lb_frac is high
    # Use 3-point interpolation: fast, mid, slow
    mid_point = 0.5
    adaptive_roc = pd.Series(
        np.where(
            lb_frac <= mid_point,
            roc_fast * (1 - lb_frac / mid_point) + roc_mid * (lb_frac / mid_point),
            roc_mid * (1 - (lb_frac - mid_point) / mid_point)
            + roc_slow * ((lb_frac - mid_point) / mid_point),
        ),
        index=df.index,
    )
    df["adaptive_roc"] = adaptive_roc

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    # --- Drawdown (HEDGE_ONLY) ---
    df["drawdown"] = drawdown(close)

    return df
