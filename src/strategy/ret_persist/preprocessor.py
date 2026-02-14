"""Return Persistence Score preprocessor.

Computes positive return bar ratio and vol-target features.
"""

from __future__ import annotations

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

    from src.strategy.ret_persist.config import RetPersistConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: RetPersistConfig) -> pd.DataFrame:
    """Return Persistence Score feature computation.

    Calculated Columns:
        - returns: Log returns
        - realized_vol: Annualized realized volatility
        - vol_scalar: Vol-target scalar
        - pos_ratio: Rolling ratio of positive return bars
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

    # --- Positive Return Ratio ---
    positive_bar = (returns > 0).astype(float)
    df["pos_ratio"] = positive_bar.rolling(
        window=config.persist_window,
        min_periods=config.persist_window,
    ).mean()

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    # --- Drawdown (HEDGE_ONLY) ---
    df["drawdown"] = drawdown(close)

    return df
