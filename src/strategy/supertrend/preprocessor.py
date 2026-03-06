"""SuperTrend 전처리 모듈.

SuperTrend + ADX 지표를 계산한다. 모든 연산은 벡터화.
"""

import pandas as pd

from src.market.indicators import adx as calc_adx, atr as calc_atr, supertrend as calc_supertrend
from src.strategy.supertrend.config import SuperTrendConfig

_REQUIRED_COLUMNS = frozenset({"high", "low", "close"})


def preprocess(df: pd.DataFrame, config: SuperTrendConfig) -> pd.DataFrame:
    """SuperTrend + ADX 지표 계산.

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: 전략 설정

    Returns:
        SuperTrend/ADX 컬럼이 추가된 새 DataFrame
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    df = df.copy()

    high: pd.Series = df["high"]  # type: ignore[assignment]
    low: pd.Series = df["low"]  # type: ignore[assignment]
    close: pd.Series = df["close"]  # type: ignore[assignment]

    st_line, st_dir = calc_supertrend(
        high,
        low,
        close,
        period=config.atr_period,
        multiplier=config.multiplier,
    )

    df["supertrend"] = st_line
    df["supertrend_dir"] = st_dir
    df["atr"] = calc_atr(high, low, close, period=config.atr_period)

    if config.use_adx_filter:
        df["adx"] = calc_adx(high, low, close, period=config.adx_period)

    return df
