"""BTC-Lead Follower Signal 전처리 모듈.

BTC 수익률 + smoothing feature 계산. btc_close 컬럼 필요.
btc_close 없으면 (BTC 자체 분석) 자기 모멘텀으로 fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.btc_lead.config import BtcLeadConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: BtcLeadConfig) -> pd.DataFrame:
    """BTC-Lead Follower Signal feature 계산.

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

    # --- BTC Returns ---
    if "btc_close" in df.columns:
        btc_close: pd.Series = df["btc_close"]  # type: ignore[assignment]
        btc_close = btc_close.ffill()
        btc_returns = np.log(btc_close / btc_close.shift(1))
    else:
        # BTC itself: use own returns (lead-lag with self = no alpha)
        btc_returns = returns

    df["btc_returns"] = btc_returns

    # --- Smoothed BTC momentum ---
    df["btc_mom_smooth"] = btc_returns.rolling(
        window=config.btc_mom_window, min_periods=config.btc_mom_window
    ).mean()

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
