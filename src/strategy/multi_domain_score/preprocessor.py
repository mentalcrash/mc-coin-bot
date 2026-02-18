"""Multi-Domain Score 전처리 모듈.

OHLCV + FR 데이터에서 4차원 스코어 feature 계산.
"""

import numpy as np
import pandas as pd

from src.market.indicators import (
    atr,
    log_returns,
    realized_volatility,
    sma,
    volatility_scalar,
)
from src.market.indicators.derivatives import funding_zscore
from src.market.indicators.oscillators import roc
from src.market.indicators.trend import efficiency_ratio
from src.market.indicators.volume import obv
from src.strategy.multi_domain_score.config import MultiDomainScoreConfig

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


def preprocess(df: pd.DataFrame, config: MultiDomainScoreConfig) -> pd.DataFrame:
    """Multi-Domain Score feature 계산.

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
    volume: pd.Series = df["volume"]  # type: ignore[assignment]

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

    # --- D1: Trend ---
    df["er"] = efficiency_ratio(close, period=config.er_window)
    sma_val = sma(close, period=config.sma_window)
    df["sma_direction"] = np.sign(close - sma_val)

    # --- D2: Volume ---
    obv_val = obv(close, volume)
    df["obv_slope"] = roc(obv_val, period=config.obv_roc_window)

    # --- D3: Derivatives ---
    fr: pd.Series = df["funding_rate"]  # type: ignore[assignment]
    df["fr_z"] = funding_zscore(
        fr,
        ma_window=config.fr_ma_window,
        zscore_window=config.fr_zscore_window,
    )

    # --- D4: Volatility ---
    rv_short = realized_volatility(
        returns,
        window=config.rv_short_window,
        annualization_factor=config.annualization_factor,
    )
    rv_long = realized_volatility(
        returns,
        window=config.rv_long_window,
        annualization_factor=config.annualization_factor,
    )
    df["rv_ratio"] = rv_short / rv_long.replace(0, float("nan"))

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    return df
