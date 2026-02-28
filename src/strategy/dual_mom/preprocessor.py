"""Dual Momentum Preprocessor (Indicator Calculation).

XSMOM preprocessor와 동일 패턴: rolling_return, realized_vol, vol_scalar, atr.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #26 VectorBT Standards: Compatible output format
"""

import logging

import pandas as pd

from src.market.indicators import (
    atr,
    log_returns,
    realized_volatility,
    rolling_return,
    simple_returns,
    volatility_scalar,
)
from src.strategy.dual_mom.config import DualMomConfig

logger = logging.getLogger(__name__)


def preprocess(
    df: pd.DataFrame,
    config: DualMomConfig,
) -> pd.DataFrame:
    """Dual Momentum 전처리 (순수 지표 계산).

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - realized_vol: 실현 변동성 (연환산)
        - rolling_return: lookback 기간 수익률
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: DualMom 설정

    Returns:
        지표가 추가된 새로운 DataFrame
    """
    required_cols = {"close", "high", "low"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    result = df.copy()

    # OHLCV 컬럼을 float64로 변환
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    close_series: pd.Series = result["close"]  # type: ignore[assignment]
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]

    # 1. 수익률
    result["returns"] = (
        log_returns(close_series) if config.use_log_returns else simple_returns(close_series)
    )
    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. 실현 변동성
    result["realized_vol"] = realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 3. Rolling return
    result["rolling_return"] = rolling_return(
        close_series,
        period=config.lookback,
        use_log=config.use_log_returns,
    )

    # 4. 변동성 스케일러
    result["vol_scalar"] = volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 5. ATR (리스크 관리용)
    result["atr"] = atr(high_series, low_series, close_series)

    return result
