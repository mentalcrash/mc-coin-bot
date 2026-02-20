"""FR Quality Momentum 전처리 모듈 (Derivatives).

OHLCV + Funding Rate 데이터에서 모멘텀 + FR crowding feature 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.fr_quality_mom.config import FrQualityMomConfig

from src.market.indicators import (
    drawdown,
    funding_rate_ma,
    funding_zscore,
    log_returns,
    realized_volatility,
    roc,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "funding_rate"})


def preprocess(df: pd.DataFrame, config: FrQualityMomConfig) -> pd.DataFrame:
    """FR Quality Momentum feature 계산 (Derivatives 포함).

    Args:
        df: OHLCV + funding_rate DataFrame
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

    # --- Momentum (ROC) ---
    df["momentum"] = roc(close, period=config.momentum_window)

    # --- Funding Rate Features ---
    funding_rate: pd.Series = df["funding_rate"]  # type: ignore[assignment]
    funding_rate = funding_rate.ffill()  # merge_asof 후 NaN 처리

    df["avg_funding_rate"] = funding_rate_ma(funding_rate, window=config.fr_lookback)
    df["fr_zscore"] = funding_zscore(
        funding_rate,
        ma_window=config.fr_lookback,
        zscore_window=config.fr_zscore_window,
    )

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
