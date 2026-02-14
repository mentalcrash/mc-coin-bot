"""Vol-of-Vol Momentum 전처리 모듈.

OHLCV 데이터에서 GK volatility, VoV, 모멘텀 feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.vov_mom.config import VovMomConfig

from src.market.indicators import (
    drawdown,
    garman_klass_volatility,
    log_returns,
    realized_volatility,
    rolling_return,
    volatility_of_volatility,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: VovMomConfig) -> pd.DataFrame:
    """Vol-of-Vol Momentum feature 계산.

    Calculated columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - gk_vol: GK realized volatility (rolling sqrt of GK variance)
        - vov: VoV (volatility of volatility)
        - vov_pct: VoV percentile rank (0~1)
        - price_mom: 가격 모멘텀
        - drawdown: rolling drawdown (HEDGE_ONLY용)

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
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
    open_: pd.Series = df["open"]  # type: ignore[assignment]

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

    # --- GK Volatility ---
    gk_var = garman_klass_volatility(open_, high, low, close)
    # Rolling mean of GK variance → sqrt for volatility
    gk_vol_sq: pd.Series = gk_var.rolling(  # type: ignore[assignment]
        window=config.gk_window, min_periods=config.gk_window
    ).mean()
    gk_vol: pd.Series = pd.Series(  # type: ignore[no-redef]
        np.sqrt(gk_vol_sq.clip(lower=0)), index=df.index
    )
    df["gk_vol"] = gk_vol

    # --- VoV (Volatility of Volatility) ---
    vov = volatility_of_volatility(gk_vol, window=config.vov_window)
    df["vov"] = vov

    # --- VoV Percentile Rank ---
    vov_pct: pd.Series = vov.rolling(  # type: ignore[assignment]
        window=config.vov_percentile_window,
        min_periods=min(config.vov_percentile_window, 60),
    ).rank(pct=True)
    df["vov_pct"] = vov_pct

    # --- Price Momentum ---
    df["price_mom"] = rolling_return(close, period=config.mom_lookback)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
