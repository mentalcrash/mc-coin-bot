"""Macro-Gated Patient Trend (4H) — preprocessor.

No shift() here — all vectorized indicator computation.
Macro columns are injected by StrategyEngine via merge_asof(direction='backward').
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.market.indicators import (
    donchian_channel,
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.macro_patience_4h.config import MacroPatience4hConfig

_REQUIRED_OHLCV = frozenset({"open", "high", "low", "close", "volume"})
_MACRO_COLUMNS = ("macro_dxy", "macro_vix", "macro_m2")


def preprocess(df: pd.DataFrame, config: MacroPatience4hConfig) -> pd.DataFrame:
    """Compute macro z-scores and multi-scale Donchian channels."""
    missing = _REQUIRED_OHLCV - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    df = df.copy()

    close: pd.Series = df["close"]  # type: ignore[assignment]
    high: pd.Series = df["high"]  # type: ignore[assignment]
    low: pd.Series = df["low"]  # type: ignore[assignment]

    # --- Multi-scale Donchian channels ---
    for scale in (config.dc_scale_short, config.dc_scale_mid, config.dc_scale_long):
        upper, _mid, lower = donchian_channel(high, low, scale)
        df[f"dc_upper_{scale}"] = upper
        df[f"dc_lower_{scale}"] = lower

    # --- Volatility & drawdown ---
    returns = log_returns(close)
    df["returns"] = returns
    rv = realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    df["realized_vol"] = rv
    df["vol_scalar"] = volatility_scalar(
        rv,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )
    df["drawdown"] = drawdown(close)

    # --- Macro z-scores ---
    df = _compute_macro_composite(df, config)

    return df


def _compute_macro_composite(df: pd.DataFrame, config: MacroPatience4hConfig) -> pd.DataFrame:
    """Compute macro direction composite z-score.

    Graceful degradation: if macro columns are absent (non-macro backtest),
    macro_direction defaults to 0 (neutral = no macro gate).
    """
    w = config.macro_z_window

    has_all_macro = all(col in df.columns for col in _MACRO_COLUMNS)
    if not has_all_macro:
        df["macro_z"] = 0.0
        df["macro_direction"] = 0
        return df

    # Forward-fill macro data (merge_asof may leave NaN at boundaries)
    for col in _MACRO_COLUMNS:
        df[col] = df[col].ffill()

    # DXY z-score (inverted: DXY down = risk-on = crypto bullish)
    dxy: pd.Series = df["macro_dxy"]  # type: ignore[assignment]
    dxy_z = _rolling_zscore(dxy, w)

    # VIX z-score (inverted: VIX down = calm = trend following works)
    vix: pd.Series = df["macro_vix"]  # type: ignore[assignment]
    vix_z = _rolling_zscore(vix, w)

    # M2 growth rate z-score (M2 up = liquidity expansion)
    m2: pd.Series = df["macro_m2"]  # type: ignore[assignment]
    m2_growth: pd.Series = m2.pct_change(config.m2_growth_window)  # type: ignore[assignment]
    m2_z = _rolling_zscore(m2_growth, w)

    # Composite z-score
    macro_z: pd.Series = (
        config.dxy_weight * dxy_z + config.vix_weight * vix_z + config.m2_weight * m2_z
    )
    df["macro_z"] = macro_z.fillna(0.0)

    # Direction: +1 (risk-on), -1 (risk-off), 0 (neutral)
    df["macro_direction"] = np.where(
        macro_z > config.macro_z_threshold,
        1,
        np.where(macro_z < -config.macro_z_threshold, -1, 0),
    )

    return df


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z-score."""
    rolling_mean = series.rolling(window, min_periods=max(1, window // 2)).mean()
    rolling_std = series.rolling(window, min_periods=max(1, window // 2)).std()
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    return ((series - rolling_mean) / rolling_std).fillna(0.0)
