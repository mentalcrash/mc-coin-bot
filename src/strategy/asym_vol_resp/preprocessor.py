"""Asymmetric Volume Response 전처리 모듈.

OHLCV에서 volume-price impact, asymmetry features 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.asym_vol_resp.config import AsymVolRespConfig

from src.market.indicators import (
    atr,
    drawdown,
    log_returns,
    realized_volatility,
    rolling_zscore,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: AsymVolRespConfig) -> pd.DataFrame:
    """Asymmetric Volume Response feature 계산.

    Calculated Features:
        - up_impact: Rolling mean of |return|/volume on up bars
        - down_impact: Rolling mean of |return|/volume on down bars
        - impact_asymmetry: up_impact - down_impact (z-scored)
        - volume_norm: Volume z-score
        - mom_return: Momentum return

    Args:
        df: OHLCV DataFrame
        config: 전략 설정

    Returns:
        feature가 추가된 새 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
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

    # --- Volume-Price Impact ---
    # Impact = |return| / volume (price impact per unit volume)
    volume_safe = volume.clip(lower=1.0)
    abs_return = returns.abs()
    impact = abs_return / volume_safe

    # Separate up and down bars
    up_mask = returns > 0
    down_mask = returns < 0

    up_impact = pd.Series(np.where(up_mask, impact, np.nan), index=df.index)
    down_impact = pd.Series(np.where(down_mask, impact, np.nan), index=df.index)

    # Rolling mean of up/down impact
    w = config.impact_window
    df["up_impact"] = up_impact.rolling(w, min_periods=max(1, w // 2)).mean()
    df["down_impact"] = down_impact.rolling(w, min_periods=max(1, w // 2)).mean()

    # --- Impact Asymmetry ---
    # Positive = up moves have more impact per unit volume = buying pressure
    # Negative = down moves have more impact = selling pressure
    raw_asym = df["up_impact"] - df["down_impact"]
    df["impact_asymmetry"] = rolling_zscore(
        raw_asym,
        window=config.asym_window,
    )

    # --- Volume Z-score (normalized) ---
    df["volume_norm"] = rolling_zscore(volume, window=config.asym_window)

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
