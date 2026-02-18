"""Adaptive FR Carry 전처리 모듈.

OHLCV + FR 데이터에서 fr_z, atr_ratio, efficiency_ratio, vol_scalar 계산.
"""

import pandas as pd

from src.market.indicators import (
    atr,
    log_returns,
    realized_volatility,
    volatility_scalar,
)
from src.market.indicators.derivatives import funding_zscore
from src.market.indicators.trend import efficiency_ratio
from src.strategy.adaptive_fr_carry.config import AdaptiveFrCarryConfig

_REQUIRED_COLUMNS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "funding_rate",
    }
)


def preprocess(df: pd.DataFrame, config: AdaptiveFrCarryConfig) -> pd.DataFrame:
    """Adaptive FR Carry feature 계산.

    Args:
        df: OHLCV + derivatives DataFrame (DatetimeIndex 필수)
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

    # --- Funding Rate Z-Score ---
    fr: pd.Series = df["funding_rate"]  # type: ignore[assignment]
    df["fr_z"] = funding_zscore(
        fr,
        ma_window=config.fr_ma_window,
        zscore_window=config.fr_zscore_window,
    )

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    # --- ATR Ratio (short/long) ---
    atr_val: pd.Series = df["atr"]  # type: ignore[assignment]
    atr_short: pd.Series = atr_val.rolling(config.atr_short_window).mean()  # type: ignore[assignment]
    atr_long: pd.Series = atr_val.rolling(config.atr_long_window).mean()  # type: ignore[assignment]
    df["atr_ratio"] = atr_short / atr_long.replace(0, float("nan"))

    # --- Efficiency Ratio ---
    df["er"] = efficiency_ratio(close, period=config.atr_long_window)

    return df
