"""Vol Squeeze Breakout 전처리 모듈.

OHLCV 데이터에서 BB width, ATR ratio, squeeze 상태, 거래량 features를 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

import numpy as np
import pandas as pd

from src.market.indicators import (
    atr,
    bollinger_bands,
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)
from src.strategy.vol_squeeze_brk.config import VolSqueezeBrkConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: VolSqueezeBrkConfig) -> pd.DataFrame:
    """Vol Squeeze Breakout feature 계산.

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

    # --- Bollinger Bands ---
    bb_upper, bb_mid, bb_lower = bollinger_bands(
        close, period=config.bb_period, std_dev=config.bb_std
    )
    df["bb_upper"] = bb_upper
    df["bb_mid"] = bb_mid
    df["bb_lower"] = bb_lower

    # --- BB Width & Percentile ---
    bb_mid_safe: pd.Series = bb_mid.replace(0, np.nan)  # type: ignore[assignment]
    bb_width = (bb_upper - bb_lower) / bb_mid_safe
    df["bb_width"] = bb_width
    df["bb_width_pct"] = bb_width.rolling(
        window=config.bb_pct_window, min_periods=config.bb_pct_window
    ).rank(pct=True)

    # --- ATR & ATR Ratio ---
    atr_val = atr(high, low, close, period=config.atr_period)
    df["atr"] = atr_val
    atr_long_avg: pd.Series = atr_val.rolling(  # type: ignore[assignment]
        window=config.atr_ratio_window, min_periods=config.atr_ratio_window
    ).mean()
    atr_long_safe: pd.Series = atr_long_avg.replace(0, np.nan)  # type: ignore[assignment]
    df["atr_ratio"] = atr_val / atr_long_safe

    # --- Squeeze Detection ---
    bb_squeeze = df["bb_width_pct"] < config.bb_pct_threshold
    atr_squeeze = df["atr_ratio"] < config.atr_ratio_threshold
    df["in_squeeze"] = bb_squeeze & atr_squeeze

    # --- Volume Average ---
    df["vol_avg"] = volume.rolling(
        window=config.vol_surge_window, min_periods=config.vol_surge_window
    ).mean()

    # --- Drawdown (HEDGE_ONLY) ---
    df["drawdown"] = drawdown(close)

    return df
