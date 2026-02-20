"""Volatility Ratio Trend 전처리 모듈.

단기/장기 RV 비율 + 모멘텀 feature 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.vol_ratio_trend.config import VolRatioTrendConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    roc,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: VolRatioTrendConfig) -> pd.DataFrame:
    """Volatility Ratio Trend feature 계산.

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

    # --- Realized Volatility (for vol_scalar) ---
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

    # --- Short-term / Long-term RV ---
    short_rv = realized_volatility(
        returns,
        window=config.short_vol_window,
        annualization_factor=config.annualization_factor,
    )
    long_rv = realized_volatility(
        returns,
        window=config.long_vol_window,
        annualization_factor=config.annualization_factor,
    )

    # --- Vol Ratio (short / long) ---
    # < 1.0 = contango (short vol < long vol, calm)
    # > 1.0 = backwardation (short vol > long vol, stress)
    raw_ratio = short_rv / long_rv.clip(lower=1e-10)
    df["vol_ratio"] = raw_ratio.rolling(
        window=config.ratio_smooth_window, min_periods=config.ratio_smooth_window
    ).mean()

    # --- Momentum (ROC) ---
    df["momentum"] = roc(close, period=config.momentum_window)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
