"""Trend Efficiency Scorer 전처리 모듈.

OHLCV 데이터에서 ER, multi-horizon ROC, ADX, vol-target feature를 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

import pandas as pd

from src.market.indicators import (
    adx,
    atr,
    drawdown,
    efficiency_ratio,
    log_returns,
    realized_volatility,
    roc,
    volatility_scalar,
)
from src.strategy.trend_eff_score.config import TrendEffScoreConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: TrendEffScoreConfig) -> pd.DataFrame:
    """Trend Efficiency Scorer feature 계산.

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
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

    # --- Efficiency Ratio ---
    df["er"] = efficiency_ratio(close, period=config.er_window)

    # --- Multi-Horizon ROC ---
    df["roc_short"] = roc(close, period=config.roc_short)
    df["roc_medium"] = roc(close, period=config.roc_medium)
    df["roc_long"] = roc(close, period=config.roc_long)

    # --- ADX ---
    df["adx_val"] = adx(high, low, close, period=config.adx_period)

    # --- Drawdown (HEDGE_ONLY) ---
    df["drawdown"] = drawdown(close)

    # --- ATR (trailing stop) ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    return df
