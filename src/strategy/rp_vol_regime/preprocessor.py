"""Realized-Parkinson Vol Regime 전처리 모듈.

RV와 PV를 계산하고 비율의 z-score로 시장 상태 식별.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.rp_vol_regime.config import RpVolRegimeConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    parkinson_volatility,
    realized_volatility,
    roc,
    volatility_scalar,
)
from src.market.indicators.composite import rolling_zscore

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: RpVolRegimeConfig) -> pd.DataFrame:
    """Realized-Parkinson Vol Regime feature 계산.

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

    # --- Returns ---
    returns = log_returns(close)
    df["returns"] = returns

    # --- Realized Volatility (close-to-close) ---
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

    # --- Realized Vol (rolling, for ratio) ---
    rv_rolling = realized_volatility(
        returns,
        window=config.rv_window,
        annualization_factor=config.annualization_factor,
    )
    df["rv"] = rv_rolling

    # --- Parkinson Volatility (intraday range-based) ---
    pv_single = parkinson_volatility(high, low)
    pv_rolling = pv_single.rolling(window=config.pv_window, min_periods=config.pv_window).mean()
    # Annualize: multiply by sqrt(annualization_factor)
    pv_annualized: pd.Series = pv_rolling * np.sqrt(config.annualization_factor)  # type: ignore[assignment]
    df["pv"] = pv_annualized

    # --- PV/RV Ratio ---
    rv_safe = rv_rolling.clip(lower=1e-10)
    df["pv_rv_ratio"] = pv_annualized / rv_safe

    # --- Z-score of PV/RV ratio ---
    pv_rv_ratio: pd.Series = df["pv_rv_ratio"]  # type: ignore[assignment]
    df["pv_rv_zscore"] = rolling_zscore(pv_rv_ratio, window=config.ratio_zscore_window)

    # --- Momentum ---
    df["momentum"] = roc(close, period=config.momentum_window)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
