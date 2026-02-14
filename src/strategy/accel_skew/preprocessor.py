"""Acceleration-Skewness Signal 전처리 모듈.

OHLCV 데이터에서 acceleration, rolling skewness, vol-target feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

import pandas as pd

from src.market.indicators import (
    atr,
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)
from src.strategy.accel_skew.config import AccelSkewConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: AccelSkewConfig) -> pd.DataFrame:
    """Acceleration-Skewness Signal feature 계산.

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

    # --- Smoothed Acceleration: rolling_mean(returns.diff(), N) ---
    acceleration = returns.diff()
    df["acceleration"] = acceleration.rolling(
        window=config.acc_smooth_window,
        min_periods=config.acc_smooth_window,
    ).mean()

    # --- Rolling Skewness ---
    df["rolling_skew"] = returns.rolling(
        window=config.skew_window,
        min_periods=config.skew_window,
    ).skew()

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    return df
