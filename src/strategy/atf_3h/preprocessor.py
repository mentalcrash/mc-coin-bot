"""Anchored Trend-Following 3H 전처리 모듈.

Anchor-Mom과 동일 로직, 3H TF 적응. Nearness + momentum direction.
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
from src.strategy.atf_3h.config import Atf3hConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: Atf3hConfig) -> pd.DataFrame:
    """Anchored Trend-Following 3H feature 계산.

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

    # --- Nearness: close / rolling_max(close, lookback) ---
    rolling_max: pd.Series = close.rolling(  # type: ignore[assignment]
        window=config.nearness_lookback,
        min_periods=config.nearness_lookback,
    ).max()
    rolling_max_safe: pd.Series = rolling_max.replace(0, np.nan)  # type: ignore[assignment]
    df["nearness"] = close / rolling_max_safe

    # --- Momentum direction: sign(close / close.shift(M) - 1) ---
    mom_return = close / close.shift(config.mom_lookback) - 1.0
    df["mom_direction"] = np.sign(mom_return)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
