"""OBV Acceleration Momentum preprocessor.

Computes OBV, smoothed OBV, velocity, acceleration, and z-scored acceleration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.market.indicators import (
    atr,
    drawdown,
    ema,
    log_returns,
    obv,
    realized_volatility,
    rolling_zscore,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.obv_accel.config import ObvAccelConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: ObvAccelConfig) -> pd.DataFrame:
    """OBV Acceleration Momentum feature computation.

    Calculated Columns:
        - returns: Log returns
        - realized_vol: Annualized realized volatility
        - vol_scalar: Vol-target scalar
        - obv_raw: Raw OBV
        - obv_smooth: EMA-smoothed OBV
        - obv_velocity: 1st derivative (diff of smoothed OBV)
        - obv_accel: 2nd derivative (diff of velocity)
        - obv_accel_z: Z-scored acceleration
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

    # --- OBV ---
    obv_raw = obv(close, volume)
    df["obv_raw"] = obv_raw

    # --- Smoothed OBV ---
    obv_smoothed = ema(obv_raw, span=config.obv_smooth)
    df["obv_smooth"] = obv_smoothed

    # --- OBV Velocity (1st derivative) ---
    obv_velocity = obv_smoothed.diff()
    df["obv_velocity"] = obv_velocity

    # --- OBV Acceleration (2nd derivative) ---
    obv_accel = obv_velocity.diff()
    df["obv_accel"] = obv_accel

    # --- Z-scored Acceleration ---
    df["obv_accel_z"] = rolling_zscore(obv_accel, window=config.accel_window + 20)

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
