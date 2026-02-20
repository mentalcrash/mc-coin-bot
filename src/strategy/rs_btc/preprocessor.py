"""Relative Strength vs BTC 전처리 모듈.

에셋 vs BTC 상대 강도(RS) feature 계산.
btc_close 없으면 RS=0 (BTC 자체 대비 상대 강도 무의미).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.rs_btc.config import RsBtcConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: RsBtcConfig) -> pd.DataFrame:
    """Relative Strength vs BTC feature 계산.

    Args:
        df: OHLCV DataFrame (+ optional btc_close column)
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

    # --- Relative Strength vs BTC ---
    if "btc_close" in df.columns:
        btc_close: pd.Series = df["btc_close"]  # type: ignore[assignment]
        btc_close = btc_close.ffill()
        btc_returns = np.log(btc_close / btc_close.shift(1))

        # RS = cumulative asset return - cumulative BTC return over window
        asset_cum = returns.rolling(window=config.rs_window, min_periods=config.rs_window).sum()
        btc_cum = btc_returns.rolling(window=config.rs_window, min_periods=config.rs_window).sum()
        rs_raw = asset_cum - btc_cum
    else:
        # BTC vs itself: RS = 0
        rs_raw = pd.Series(0.0, index=df.index)

    df["relative_strength"] = rs_raw

    # --- Smoothed RS ---
    df["rs_smooth"] = rs_raw.rolling(
        window=config.rs_smooth_window, min_periods=config.rs_smooth_window
    ).mean()

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
