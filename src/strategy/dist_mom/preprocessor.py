"""Return Distribution Momentum 전처리 모듈.

수익률 분포 특성(양수비율, skewness) + 모멘텀 feature 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.dist_mom.config import DistMomConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    roc,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: DistMomConfig) -> pd.DataFrame:
    """Return Distribution Momentum feature 계산.

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

    # --- Positive Day Proportion (핵심 feature) ---
    positive_returns = (returns > 0).astype(float)
    df["pos_day_ratio"] = positive_returns.rolling(
        window=config.dist_window, min_periods=config.dist_window
    ).mean()

    # --- Return Skewness ---
    df["return_skew"] = returns.rolling(
        window=config.skew_window, min_periods=config.skew_window
    ).skew()

    # --- Momentum (ROC) ---
    df["momentum"] = roc(close, period=config.momentum_window)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
