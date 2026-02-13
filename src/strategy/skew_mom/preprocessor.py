"""Skew-Gated Momentum 전처리 모듈.

OHLCV 데이터에서 rolling skewness, momentum, vol-target feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

import numpy as np
import pandas as pd

from src.strategy.skew_mom.config import SkewMomConfig
from src.strategy.tsmom.preprocessor import (
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: SkewMomConfig) -> pd.DataFrame:
    """Skew-Gated Momentum feature 계산.

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

    # --- Returns ---
    returns = calculate_returns(close)
    df["returns"] = returns

    # --- Realized Volatility ---
    realized_vol = calculate_realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    df["realized_vol"] = realized_vol

    # --- Vol Scalar ---
    df["vol_scalar"] = calculate_volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # --- Rolling Skewness ---
    df["skewness"] = returns.rolling(
        window=config.skew_window, min_periods=config.skew_window
    ).skew()

    # --- Momentum direction: sign of cumulative return over lookback ---
    mom_return = close / close.shift(config.mom_lookback) - 1.0
    df["mom_direction"] = np.sign(mom_return)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = calculate_drawdown(close)

    return df
