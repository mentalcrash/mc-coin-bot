"""R2 Consensus Trend 전처리 모듈.

3개 스케일(short/mid/long)의 rolling OLS로 slope과 R^2를 계산하고,
vol-target / drawdown feature를 생성한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.market.indicators import (
    atr,
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    from src.strategy.r2_consensus.config import R2ConsensusConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def _rolling_regression(
    close: pd.Series,
    lookback: int,
) -> tuple[pd.Series, pd.Series]:
    """Compute rolling OLS slope and R-squared on close prices.

    Uses rolling.apply with numpy for vectorized-style computation.
    R^2 measures the 'quality' (linearity) of the trend in the lookback window.

    Args:
        close: Close price series.
        lookback: Rolling window size.

    Returns:
        Tuple of (slope_series, r2_series).
    """
    x = np.arange(lookback, dtype=float)
    x_mean = x.mean()
    ss_xx = float(np.sum((x - x_mean) ** 2))
    _ss_epsilon = 1e-15  # Minimum SS_yy to avoid near-zero division

    def _r2_apply(window_data: np.ndarray) -> float:  # type: ignore[type-arg]
        y = window_data.astype(float)
        y_mean = y.mean()
        ss_yy = float(np.sum((y - y_mean) ** 2))
        if ss_yy < _ss_epsilon:
            return 0.0
        ss_xy = float(np.sum((x - x_mean) * (y - y_mean)))
        r_squared = (ss_xy**2) / (ss_xx * ss_yy)
        return float(np.clip(r_squared, 0.0, 1.0))

    def _slope_apply(window_data: np.ndarray) -> float:  # type: ignore[type-arg]
        y = window_data.astype(float)
        y_mean = y.mean()
        ss_xy = float(np.sum((x - x_mean) * (y - y_mean)))
        return ss_xy / ss_xx if ss_xx > 0 else 0.0

    r2_series: pd.Series = close.rolling(  # type: ignore[assignment]
        window=lookback,
        min_periods=lookback,
    ).apply(_r2_apply, raw=True)

    slope_series: pd.Series = close.rolling(  # type: ignore[assignment]
        window=lookback,
        min_periods=lookback,
    ).apply(_slope_apply, raw=True)

    return (
        pd.Series(slope_series, index=close.index),
        pd.Series(r2_series, index=close.index),
    )


def preprocess(df: pd.DataFrame, config: R2ConsensusConfig) -> pd.DataFrame:
    """R2 Consensus Trend feature 계산.

    Calculated Columns:
        - returns: Log returns
        - realized_vol: Annualized realized volatility
        - vol_scalar: Vol-target scalar
        - slope_{lb}, r2_{lb}: 각 스케일 rolling OLS slope 및 R^2
        - atr: Average True Range
        - drawdown: Peak drawdown (for HEDGE_ONLY)

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

    # --- Multi-Scale Linear Regression: Slope + R^2 ---
    lookbacks = (config.lookback_short, config.lookback_mid, config.lookback_long)
    for lb in lookbacks:
        slope, r2 = _rolling_regression(close, lb)
        df[f"slope_{lb}"] = slope
        df[f"r2_{lb}"] = r2

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    # --- Drawdown (HEDGE_ONLY) ---
    df["drawdown"] = drawdown(close)

    return df
