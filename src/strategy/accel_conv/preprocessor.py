"""Acceleration-Conviction Momentum 전처리 모듈.

OHLCV 데이터에서 acceleration, conviction, vol-target feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

import numpy as np
import pandas as pd

from src.market.indicators import (
    atr,
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)
from src.strategy.accel_conv.config import AccelConvConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: AccelConvConfig) -> pd.DataFrame:
    """Acceleration-Conviction Momentum feature 계산.

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

    # --- Acceleration: smoothed 2nd derivative of returns ---
    acceleration = returns.diff()
    df["acceleration"] = acceleration.rolling(
        window=config.smooth_window,
        min_periods=config.smooth_window,
    ).mean()

    # --- Conviction: abs(close - open) / (high - low) ---
    range_ = high - low
    range_safe: pd.Series = range_.replace(0, np.nan)  # type: ignore[assignment]
    raw_conviction = (close - open_).abs() / range_safe
    df["conviction"] = raw_conviction.rolling(
        window=config.smooth_window,
        min_periods=config.smooth_window,
    ).mean()

    # --- Composite signal: sign(acceleration) * conviction ---
    acc_series: pd.Series = df["acceleration"]  # type: ignore[assignment]
    conv_series: pd.Series = df["conviction"]  # type: ignore[assignment]
    df["composite_signal"] = np.sign(acc_series) * conv_series

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    return df
