"""Momentum Acceleration 전처리 모듈.

모멘텀 속도(ROC) + 가속도(price_acceleration) feature 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.mom_accel.config import MomAccelConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    price_acceleration,
    realized_volatility,
    roc,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: MomAccelConfig) -> pd.DataFrame:
    """Momentum Acceleration feature 계산.

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

    # --- Momentum (속도, 1st derivative) ---
    df["momentum"] = roc(close, period=config.momentum_window)

    # --- Acceleration (2nd derivative) ---
    # price_acceleration = fast_roc - slow_roc (가속 시 양수)
    raw_accel = price_acceleration(close, fast=config.fast_roc, slow=config.slow_roc)
    df["acceleration"] = raw_accel.rolling(
        window=config.accel_window, min_periods=config.accel_window
    ).mean()

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
