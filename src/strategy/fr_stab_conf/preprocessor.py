"""Funding Rate + Stablecoin Confluence 전처리 모듈.

OHLCV + derivatives + on-chain 데이터에서 FR z-score, stablecoin z-score,
vol-target feature를 계산한다.
"""

import pandas as pd

from src.market.indicators import (
    atr,
    log_returns,
    realized_volatility,
    volatility_scalar,
)
from src.market.indicators.composite import rolling_zscore
from src.market.indicators.derivatives import funding_zscore
from src.strategy.fr_stab_conf.config import FrStabConfConfig

_REQUIRED_COLUMNS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "funding_rate",
        "oc_stablecoin_total_circulating_usd",
    }
)


def preprocess(df: pd.DataFrame, config: FrStabConfConfig) -> pd.DataFrame:
    """FR + Stablecoin Confluence feature 계산.

    Args:
        df: OHLCV + derivatives + on-chain DataFrame (DatetimeIndex 필수)
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

    # --- Stablecoin Z-Score ---
    stab_col: pd.Series = df["oc_stablecoin_total_circulating_usd"]  # type: ignore[assignment]
    stab_change = stab_col.pct_change(config.stab_change_period)
    df["stab_z"] = rolling_zscore(stab_change, window=config.stab_zscore_window)

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    return df
