"""Scaled Momentum 전처리 모듈.

OHLCV 데이터에서 (close - SMA) / ATR 정규화 모멘텀 feature를 계산.
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
from src.strategy.scaled_mom.config import ScaledMomConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: ScaledMomConfig) -> pd.DataFrame:
    """Scaled Momentum feature 계산.

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

    # --- SMA ---
    sma = close.rolling(window=config.sma_period, min_periods=config.sma_period).mean()

    # --- ATR ---
    atr_val = atr(high, low, close, period=config.atr_period)

    # --- Scaled Momentum: (close - SMA) / ATR ---
    atr_safe = atr_val.clip(lower=1e-10)
    df["scaled_mom"] = (close - sma) / atr_safe

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
