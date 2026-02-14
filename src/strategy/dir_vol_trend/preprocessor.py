"""Directional Volume Trend 전처리 모듈.

OHLCV 데이터에서 up-bar/down-bar volume ratio feature를 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

import numpy as np
import pandas as pd

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)
from src.strategy.dir_vol_trend.config import DirVolTrendConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: DirVolTrendConfig) -> pd.DataFrame:
    """Directional Volume Trend feature 계산.

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

    # --- Up/Down Volume ---
    # Up bar: close > open (bullish), Down bar: close < open (bearish)
    open_: pd.Series = df["open"]  # type: ignore[assignment]
    is_up = close > open_
    is_down = close < open_

    up_volume = pd.Series(np.where(is_up, volume, 0.0), index=df.index)
    down_volume = pd.Series(np.where(is_down, volume, 0.0), index=df.index)

    # Rolling sum
    up_vol_sum = up_volume.rolling(window=config.dvt_window, min_periods=config.dvt_window).sum()
    down_vol_sum = down_volume.rolling(
        window=config.dvt_window, min_periods=config.dvt_window
    ).sum()

    # --- Volume Ratio: up_volume / down_volume ---
    down_vol_safe: pd.Series = down_vol_sum.clip(lower=1.0)  # type: ignore[assignment]
    vol_ratio = pd.Series(up_vol_sum / down_vol_safe, index=df.index)

    # --- Smoothing ---
    df["vol_ratio"] = vol_ratio.ewm(span=config.dvt_smooth, adjust=False).mean()

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
