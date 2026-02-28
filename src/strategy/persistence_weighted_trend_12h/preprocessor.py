"""Persistence-Weighted-Trend 12H preprocessor.

ER x (2-FD) x TrendStrength = persistence score.
3-scale 앙상블로 파라미터 로버스트니스 확보.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.market.indicators import (
    adx,
    drawdown,
    efficiency_ratio,
    fractal_dimension,
    log_returns,
    realized_volatility,
    roc,
    trend_strength,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.persistence_weighted_trend_12h.config import PersistenceWeightedTrendConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: PersistenceWeightedTrendConfig) -> pd.DataFrame:
    """모든 feature 계산.

    Args:
        df: OHLCV DataFrame.
        config: 전략 설정.

    Returns:
        Feature 컬럼이 추가된 DataFrame.
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

    # --- Realized Volatility + Vol Scalar ---
    realized_vol = realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    df["realized_vol"] = realized_vol
    df["vol_scalar"] = volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # --- Multi-scale persistence score ---
    scales = (config.scale_short, config.scale_mid, config.scale_long)

    for s in scales:
        # Efficiency Ratio: 0~1, 높을수록 directional
        er = efficiency_ratio(close, period=s)

        # Fractal Dimension: 1~2, 1에 가까울수록 trending
        fd = fractal_dimension(close, period=s)
        # (2 - FD) 변환: 0~1, 높을수록 trending
        fd_score = (2.0 - fd).clip(0.0, 1.0)

        # Trend Strength: ADX → 0/1 판정 (threshold 25)
        adx_s = adx(high, low, close, period=s)
        ts = trend_strength(adx_s, threshold=25.0)

        # Persistence score = ER x (2-FD) x TrendStrength
        persistence = er * fd_score * ts

        df[f"persistence_{s}"] = persistence

    # --- ROC 모멘텀 방향 ---
    df["roc_direction"] = np.sign(roc(close, period=config.mom_lookback))

    # --- Drawdown for HEDGE_ONLY ---
    df["drawdown"] = drawdown(close)

    return df
