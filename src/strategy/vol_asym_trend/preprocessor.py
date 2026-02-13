"""Volatility Asymmetry Trend 전처리 모듈.

OHLCV 데이터에서 up/down semivariance 비율 feature를 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

import numpy as np
import pandas as pd

from src.strategy.tsmom.preprocessor import (
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)
from src.strategy.vol_asym_trend.config import VolAsymTrendConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: VolAsymTrendConfig) -> pd.DataFrame:
    """Volatility Asymmetry Trend feature 계산.

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

    # --- Up/Down Semivariance ---
    up_returns = returns.clip(lower=0)
    down_returns = returns.clip(upper=0)

    up_var = (
        (up_returns**2).rolling(window=config.asym_window, min_periods=config.asym_window).mean()
    )
    down_var = (
        (down_returns**2).rolling(window=config.asym_window, min_periods=config.asym_window).mean()
    )

    # --- Asymmetry Ratio: sqrt(up_var) / sqrt(down_var) ---
    up_vol = np.sqrt(up_var)
    down_vol_safe = np.sqrt(down_var).clip(lower=1e-10)
    asym_ratio: pd.Series = up_vol / down_vol_safe  # type: ignore[assignment]

    # --- Smoothing ---
    df["asym_ratio"] = asym_ratio.ewm(span=config.asym_smooth, adjust=False).mean()

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = calculate_drawdown(close)

    return df
