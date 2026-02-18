"""Fear-Greed Divergence 전처리 모듈.

OHLCV + F&G 데이터에서 fg_ma, price_roc, er, vol_scalar 계산.
"""

import pandas as pd

from src.market.indicators import (
    atr,
    log_returns,
    realized_volatility,
    sma,
    volatility_scalar,
)
from src.market.indicators.oscillators import roc
from src.market.indicators.trend import efficiency_ratio
from src.strategy.fear_divergence.config import FearDivergenceConfig

_REQUIRED_COLUMNS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "oc_fear_greed",
    }
)


def preprocess(df: pd.DataFrame, config: FearDivergenceConfig) -> pd.DataFrame:
    """Fear-Greed Divergence feature 계산.

    Args:
        df: OHLCV + on-chain (oc_fear_greed) DataFrame (DatetimeIndex 필수)
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
    fg: pd.Series = df["oc_fear_greed"]  # type: ignore[assignment]

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

    # --- F&G MA ---
    df["fg_ma"] = sma(fg, period=config.fg_ma_window)

    # --- Price ROC ---
    df["price_roc"] = roc(close, period=config.price_roc_window)

    # --- Efficiency Ratio ---
    df["er"] = efficiency_ratio(close, period=config.er_window)

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    return df
