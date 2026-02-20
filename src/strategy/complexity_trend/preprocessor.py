"""Complexity-Filtered Trend 전처리 모듈.

OHLCV 데이터에서 Hurst exponent, fractal dimension, efficiency ratio feature 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.complexity_trend.config import ComplexityTrendConfig

from src.market.indicators import (
    drawdown,
    efficiency_ratio,
    fractal_dimension,
    hurst_exponent,
    log_returns,
    realized_volatility,
    roc,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: ComplexityTrendConfig) -> pd.DataFrame:
    """Complexity-Filtered Trend feature 계산.

    Hurst exponent, fractal dimension, efficiency ratio 세 가지
    정보이론 지표를 계산하여 시장 복잡도를 정량화한다.

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

    # --- Hurst Exponent ---
    # H > 0.5: trending, H < 0.5: mean-reverting, H = 0.5: random walk
    df["hurst"] = hurst_exponent(close, window=config.hurst_window)

    # --- Fractal Dimension ---
    # D ~ 1: smooth trend, D ~ 1.5: random, D ~ 2: complex/choppy
    df["fractal_dim"] = fractal_dimension(close, period=config.fractal_period)

    # --- Efficiency Ratio ---
    # ER = net movement / total path length (0~1, 1 = perfectly efficient trend)
    df["efficiency"] = efficiency_ratio(close, period=config.er_period)

    # --- Trend Direction (SMA 기반) ---
    df["trend_sma"] = close.rolling(
        window=config.trend_window, min_periods=config.trend_window
    ).mean()

    # --- ROC for trend confirmation ---
    df["trend_roc"] = roc(close, period=config.trend_window)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
