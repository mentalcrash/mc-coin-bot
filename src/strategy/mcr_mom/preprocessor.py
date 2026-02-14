"""Momentum Crash Filter preprocessor.

Computes momentum, VoV, VoV rank, and vol-target features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.market.indicators import (
    atr,
    drawdown,
    log_returns,
    realized_volatility,
    volatility_of_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.mcr_mom.config import McrMomConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: McrMomConfig) -> pd.DataFrame:
    """Momentum Crash Filter feature computation.

    Calculated Columns:
        - returns: Log returns
        - realized_vol: Annualized realized volatility
        - vol_scalar: Vol-target scalar
        - mom_return: Rolling momentum return
        - vov: Volatility-of-volatility
        - vov_rank: VoV percentile rank (0~1)
        - crash_filter: Boolean crash filter (True = safe to trade)
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

    # --- Momentum Return ---
    df["mom_return"] = returns.rolling(
        window=config.mom_lookback,
        min_periods=config.mom_lookback,
    ).sum()

    # --- Volatility of Volatility ---
    vov = volatility_of_volatility(realized_vol, window=config.vov_window)
    df["vov"] = vov

    # --- VoV Percentile Rank (rolling 60-bar) ---
    vov_rank_window = 60
    vov_rank = vov.rolling(vov_rank_window, min_periods=vov_rank_window).rank(pct=True)
    df["vov_rank"] = vov_rank

    # --- Crash Filter: safe to trade when VoV rank is below threshold ---
    df["crash_filter"] = vov_rank <= config.vov_crash_threshold

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    # --- Drawdown (HEDGE_ONLY) ---
    df["drawdown"] = drawdown(close)

    return df
