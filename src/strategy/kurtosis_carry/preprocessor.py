"""Kurtosis Carry 전처리 모듈.

OHLCV 데이터에서 rolling kurtosis, kurtosis z-score, momentum을 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

import numpy as np
import pandas as pd

from src.market.indicators import (
    atr,
    drawdown,
    log_returns,
    realized_volatility,
    rolling_zscore,
    volatility_scalar,
)
from src.strategy.kurtosis_carry.config import KurtosisCarryConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: KurtosisCarryConfig) -> pd.DataFrame:
    """Kurtosis Carry feature 계산.

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

    # --- Rolling Kurtosis (short window) ---
    # pandas rolling.kurt() returns excess kurtosis (Fisher's definition)
    kurtosis_short: pd.Series = returns.rolling(  # type: ignore[assignment]
        window=config.kurtosis_window,
        min_periods=config.kurtosis_window,
    ).kurt()
    df["kurtosis_short"] = kurtosis_short

    # --- Rolling Kurtosis (long window) ---
    kurtosis_long: pd.Series = returns.rolling(  # type: ignore[assignment]
        window=config.kurtosis_long_window,
        min_periods=config.kurtosis_long_window,
    ).kurt()
    df["kurtosis_long"] = kurtosis_long

    # --- Kurtosis change: short - long (positive = fatter tails recently) ---
    df["kurtosis_delta"] = kurtosis_short - kurtosis_long

    # --- Z-score of kurtosis_delta ---
    df["kurtosis_zscore"] = rolling_zscore(
        kurtosis_short,
        window=config.zscore_window,
    )

    # --- Momentum confirmation ---
    mom_return = close / close.shift(config.momentum_lookback) - 1.0
    df["momentum"] = np.sign(mom_return)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    return df
