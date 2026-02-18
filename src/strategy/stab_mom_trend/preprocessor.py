"""Stablecoin Momentum Trend 전처리 모듈.

OHLCV + on-chain 데이터에서 stablecoin z-score, EMA, vol-target feature를 계산한다.
"""

import pandas as pd

from src.market.indicators import (
    atr,
    log_returns,
    realized_volatility,
    volatility_scalar,
)
from src.market.indicators.composite import rolling_zscore
from src.market.indicators.trend import ema
from src.strategy.stab_mom_trend.config import StabMomTrendConfig

_REQUIRED_COLUMNS = frozenset(
    {"open", "high", "low", "close", "volume", "oc_stablecoin_total_circulating_usd"}
)


def preprocess(df: pd.DataFrame, config: StabMomTrendConfig) -> pd.DataFrame:
    """Stablecoin Momentum Trend feature 계산.

    Args:
        df: OHLCV + on-chain DataFrame (DatetimeIndex 필수)
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

    # --- Stablecoin Z-Score ---
    stab_col: pd.Series = df["oc_stablecoin_total_circulating_usd"]  # type: ignore[assignment]
    stab_change = stab_col.pct_change(config.stab_change_period)
    df["stab_z"] = rolling_zscore(stab_change, window=config.zscore_window)

    # --- EMA Fast / Slow ---
    df["ema_fast"] = ema(close, span=config.ema_fast_period)
    df["ema_slow"] = ema(close, span=config.ema_slow_period)

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    return df
