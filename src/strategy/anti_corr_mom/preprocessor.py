"""Anti-Correlation Momentum 전처리 모듈.

에셋-BTC rolling 상관 + 모멘텀 feature 계산.
btc_close 컬럼이 없으면(BTC 자체 분석 시) 자기상관 fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.anti_corr_mom.config import AntiCorrMomConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    roc,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: AntiCorrMomConfig) -> pd.DataFrame:
    """Anti-Correlation Momentum feature 계산.

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

    # --- Asset-BTC Correlation ---
    if "btc_close" in df.columns:
        btc_close: pd.Series = df["btc_close"]  # type: ignore[assignment]
        btc_close = btc_close.ffill()
        btc_returns = np.log(btc_close / btc_close.shift(1))
        df["asset_btc_corr"] = returns.rolling(window=config.corr_window).corr(btc_returns)
    else:
        # BTC itself: use lagged autocorrelation as proxy (always high self-corr)
        df["asset_btc_corr"] = pd.Series(1.0, index=df.index)

    # --- Momentum ---
    df["momentum"] = roc(close, period=config.momentum_window)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
