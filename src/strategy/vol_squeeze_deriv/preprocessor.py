"""Vol Squeeze + Derivatives 전처리 모듈.

OHLCV + FR 데이터에서 vol_rank, atr_ratio, fr_z, sma_direction 계산.
"""

import pandas as pd

from src.market.indicators import (
    atr,
    log_returns,
    realized_volatility,
    sma,
    volatility_scalar,
)
from src.market.indicators.derivatives import funding_zscore
from src.market.indicators.volatility import vol_percentile_rank
from src.strategy.vol_squeeze_deriv.config import VolSqueezeDerivConfig

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


def preprocess(df: pd.DataFrame, config: VolSqueezeDerivConfig) -> pd.DataFrame:
    """Vol Squeeze + Derivatives feature 계산.

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

    # --- Vol Percentile Rank ---
    df["vol_rank"] = vol_percentile_rank(realized_vol, window=config.vol_rank_window)

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    # --- ATR Ratio (expansion detection) ---
    atr_val: pd.Series = df["atr"]  # type: ignore[assignment]
    atr_short: pd.Series = atr_val.rolling(config.atr_short_window).mean()  # type: ignore[assignment]
    atr_long: pd.Series = atr_val.rolling(config.atr_long_window).mean()  # type: ignore[assignment]
    df["atr_ratio"] = atr_short / atr_long.replace(0, float("nan"))

    # --- FR Z-Score ---
    fr: pd.Series = df["funding_rate"]  # type: ignore[assignment]
    df["fr_z"] = funding_zscore(
        fr,
        ma_window=config.fr_ma_window,
        zscore_window=config.fr_zscore_window,
    )

    # --- SMA Direction ---
    sma_val = sma(close, period=config.sma_direction_window)
    df["sma_direction"] = (close > sma_val).astype(int) * 2 - 1  # +1 or -1

    return df
