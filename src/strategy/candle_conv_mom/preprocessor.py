"""Candle Conviction Momentum 전처리 모듈.

OHLCV 데이터에서 candle body ratio 기반 conviction feature를 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

import numpy as np
import pandas as pd

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)
from src.strategy.candle_conv_mom.config import CandleConvMomConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: CandleConvMomConfig) -> pd.DataFrame:
    """Candle Conviction Momentum feature 계산.

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
    open_: pd.Series = df["open"]  # type: ignore[assignment]
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

    # --- Body Ratio: |close - open| / (high - low) ---
    candle_range = high - low
    body = (close - open_).abs()
    # Doji protection: range == 0 -> body_ratio = 0
    body_ratio = pd.Series(
        np.where(candle_range > 0, body / candle_range, 0.0),
        index=df.index,
    )

    # --- Candle Direction: +1 if close > open, -1 if close < open, 0 if doji ---
    candle_dir = np.sign(close - open_)

    # --- Per-bar Conviction: direction * body_ratio ---
    bar_conviction = candle_dir * body_ratio

    # --- Rolling Conviction (mean over window) ---
    df["conviction"] = bar_conviction.rolling(
        window=config.conv_window, min_periods=config.conv_window
    ).mean()

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
