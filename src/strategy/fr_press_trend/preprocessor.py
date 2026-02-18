"""Funding Pressure Trend 전처리 모듈.

OHLCV + funding_rate 데이터에서 SMA cross, ER, FR features를 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

import pandas as pd

from src.market.indicators import (
    atr,
    drawdown,
    efficiency_ratio,
    funding_rate_ma,
    funding_zscore,
    log_returns,
    realized_volatility,
    sma,
    volatility_scalar,
)
from src.strategy.fr_press_trend.config import FrPressTrendConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "funding_rate"})


def preprocess(df: pd.DataFrame, config: FrPressTrendConfig) -> pd.DataFrame:
    """Funding Pressure Trend feature 계산.

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
    high: pd.Series = df["high"]  # type: ignore[assignment]
    low: pd.Series = df["low"]  # type: ignore[assignment]
    funding_rate: pd.Series = df["funding_rate"]  # type: ignore[assignment]

    # NaN from merge_asof -> ffill
    funding_rate = funding_rate.ffill()
    df["funding_rate"] = funding_rate

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

    # --- SMA Cross ---
    df["sma_fast"] = sma(close, period=config.sma_fast)
    df["sma_slow"] = sma(close, period=config.sma_slow)

    # --- Efficiency Ratio ---
    df["er"] = efficiency_ratio(close, period=config.er_window)

    # --- Funding Rate Features ---
    df["avg_fr"] = funding_rate_ma(funding_rate, window=config.fr_ma_window)
    df["fr_z"] = funding_zscore(
        funding_rate,
        ma_window=config.fr_ma_window,
        zscore_window=config.fr_zscore_window,
    )

    # --- Drawdown (HEDGE_ONLY) ---
    df["drawdown"] = drawdown(close)

    # --- ATR (trailing stop) ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    return df
