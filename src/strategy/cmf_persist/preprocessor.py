"""CMF Trend Persistence preprocessor.

Computes CMF, sign persistence ratio, and vol-target features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.market.indicators import (
    atr,
    chaikin_money_flow,
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.cmf_persist.config import CmfPersistConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: CmfPersistConfig) -> pd.DataFrame:
    """CMF Trend Persistence feature computation.

    Calculated Columns:
        - returns: Log returns
        - realized_vol: Annualized realized volatility
        - vol_scalar: Vol-target scalar
        - cmf: Chaikin Money Flow
        - cmf_pos_ratio: Rolling ratio of CMF > 0 bars
        - cmf_neg_ratio: Rolling ratio of CMF < 0 bars
        - cmf_avg: Rolling mean CMF (conviction)
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

    # --- CMF ---
    cmf = chaikin_money_flow(high, low, close, volume, period=config.cmf_period)
    df["cmf"] = cmf

    # --- CMF Sign Persistence ---
    cmf_positive = (cmf > 0).astype(float)
    cmf_negative = (cmf < 0).astype(float)
    df["cmf_pos_ratio"] = cmf_positive.rolling(
        window=config.persist_window,
        min_periods=config.persist_window,
    ).mean()
    df["cmf_neg_ratio"] = cmf_negative.rolling(
        window=config.persist_window,
        min_periods=config.persist_window,
    ).mean()

    # --- CMF Average (conviction measure) ---
    df["cmf_avg"] = cmf.rolling(
        window=config.persist_window,
        min_periods=config.persist_window,
    ).mean()

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    # --- Drawdown (HEDGE_ONLY) ---
    df["drawdown"] = drawdown(close)

    return df
