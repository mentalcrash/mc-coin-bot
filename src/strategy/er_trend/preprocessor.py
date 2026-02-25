"""ER Trend 전처리 모듈.

Multi-lookback Signed ER feature 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.er_trend.config import ErTrendConfig

from src.market.indicators import (
    atr,
    drawdown,
    efficiency_ratio,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def _signed_er(close: pd.Series, period: int) -> pd.Series:
    """Signed Efficiency Ratio: ER * sign(direction).

    표준 ER (0~1)에 가격 변동 방향(sign)을 곱하여
    추세 품질과 방향을 동시에 표현한다. 범위: [-1, +1].

    Args:
        close: 종가 시리즈.
        period: ER 계산 기간.

    Returns:
        Signed ER 시리즈 (-1 ~ +1).
    """
    er = efficiency_ratio(close, period=period)
    direction = close - close.shift(period)
    sign = np.sign(direction)
    return pd.Series(er * sign, index=close.index, name=f"signed_er_{period}")


def preprocess(df: pd.DataFrame, config: ErTrendConfig) -> pd.DataFrame:
    """ER Trend feature 계산.

    Calculated Columns:
        - returns: Log returns
        - realized_vol: Annualized realized volatility
        - vol_scalar: Vol-target scalar
        - signed_er_fast: 단기 Signed ER
        - signed_er_mid: 중기 Signed ER
        - signed_er_slow: 장기 Signed ER
        - composite_ser: 가중 합성 Signed ER
        - atr: Average True Range
        - drawdown: Peak drawdown

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

    # --- Multi-lookback Signed ER ---
    ser_fast = _signed_er(close, period=config.er_fast)
    ser_mid = _signed_er(close, period=config.er_mid)
    ser_slow = _signed_er(close, period=config.er_slow)

    df["signed_er_fast"] = ser_fast
    df["signed_er_mid"] = ser_mid
    df["signed_er_slow"] = ser_slow

    # --- Composite Signed ER (가중 합성) ---
    composite: pd.Series = (
        config.w_fast * ser_fast + config.w_mid * ser_mid + config.w_slow * ser_slow
    )
    df["composite_ser"] = composite

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
