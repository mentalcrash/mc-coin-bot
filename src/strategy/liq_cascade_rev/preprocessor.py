"""Liquidation Cascade Reversal 전처리 모듈.

OHLCV + FR 데이터에서 fr_z, rv_ratio, body_recovery, vol_scalar 계산.
"""

import numpy as np
import pandas as pd

from src.market.indicators import (
    atr,
    log_returns,
    realized_volatility,
    volatility_scalar,
)
from src.market.indicators.derivatives import funding_zscore
from src.strategy.liq_cascade_rev.config import LiqCascadeRevConfig

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


def preprocess(df: pd.DataFrame, config: LiqCascadeRevConfig) -> pd.DataFrame:
    """Liquidation Cascade Reversal feature 계산.

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
    open_: pd.Series = df["open"]  # type: ignore[assignment]

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

    # --- FR Z-Score ---
    fr: pd.Series = df["funding_rate"]  # type: ignore[assignment]
    df["fr_z"] = funding_zscore(
        fr,
        ma_window=config.fr_ma_window,
        zscore_window=config.fr_zscore_window,
    )

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    # --- RV Ratio (short/long) ---
    rv_short = realized_volatility(
        returns,
        window=config.vol_short_window,
        annualization_factor=config.annualization_factor,
    )
    rv_long = realized_volatility(
        returns,
        window=config.vol_long_window,
        annualization_factor=config.annualization_factor,
    )
    df["rv_ratio"] = rv_short / rv_long.replace(0, float("nan"))

    # --- Body Recovery ---
    # body / (high - low) ratio: 회복 강도
    bar_range = high - low
    body = (close - open_).abs()
    df["body_recovery"] = body / bar_range.replace(0, float("nan"))

    # --- Return / ATR Ratio ---
    atr_val: pd.Series = df["atr"]  # type: ignore[assignment]
    df["return_atr_ratio"] = (close - close.shift(1)).abs() / atr_val.replace(0, float("nan"))

    # --- Return Direction ---
    return_sign = pd.Series(np.sign(returns), index=df.index)
    df["return_dir"] = return_sign.fillna(0).astype(int)

    return df
