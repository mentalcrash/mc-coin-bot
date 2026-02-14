"""Fractal-Filtered Momentum 전처리 모듈.

OHLCV에서 fractal dimension, momentum, efficiency ratio features 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.fractal_mom.config import FractalMomConfig

from src.market.indicators import (
    atr,
    drawdown,
    efficiency_ratio,
    fractal_dimension,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: FractalMomConfig) -> pd.DataFrame:
    """Fractal-Filtered Momentum feature 계산.

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

    # --- Fractal Dimension ---
    df["fractal_dim"] = fractal_dimension(close, period=config.fractal_period)

    # --- Momentum (fast & slow cumulative returns) ---
    df["mom_fast"] = returns.rolling(
        window=config.mom_fast,
        min_periods=config.mom_fast,
    ).sum()
    df["mom_slow"] = returns.rolling(
        window=config.mom_slow,
        min_periods=config.mom_slow,
    ).sum()

    # --- Efficiency Ratio ---
    df["er"] = efficiency_ratio(close, period=config.er_period)

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    # --- Drawdown (HEDGE_ONLY) ---
    df["drawdown"] = drawdown(close)

    return df
