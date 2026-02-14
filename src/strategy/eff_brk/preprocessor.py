"""Efficiency Breakout preprocessor.

Computes Kaufman Efficiency Ratio, momentum, and vol-target features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.market.indicators import (
    atr,
    drawdown,
    efficiency_ratio,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.eff_brk.config import EffBrkConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: EffBrkConfig) -> pd.DataFrame:
    """Efficiency Breakout feature computation.

    Calculated Columns:
        - returns: Log returns
        - realized_vol: Annualized realized volatility
        - vol_scalar: Vol-target scalar
        - er: Kaufman Efficiency Ratio (0~1)
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

    # --- Efficiency Ratio ---
    df["er"] = efficiency_ratio(close, period=config.er_period)

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
