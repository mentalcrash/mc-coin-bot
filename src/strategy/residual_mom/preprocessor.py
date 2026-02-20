"""Residual Momentum 전처리 모듈.

Rolling OLS로 시장 factor(EWM market proxy)에 대해 회귀 후
잔차를 추출하고, 잔차 모멘텀 및 z-score를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.residual_mom.config import ResidualMomConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def _rolling_ols_residual(
    y: pd.Series,
    x: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling OLS 잔차 계산 (벡터화).

    y = alpha + beta * x + residual 에서 residual을 rolling window로 계산.
    numpy 벡터 연산으로 구현 (루프 없음).

    Args:
        y: 종속 변수 (자산 수익률)
        x: 독립 변수 (시장 수익률)
        window: rolling window 크기

    Returns:
        잔차 Series (warmup 구간은 NaN)
    """
    # Rolling covariance / variance 로 beta 계산
    xy_cov = x.rolling(window=window, min_periods=window).cov(y)
    x_var: pd.Series = x.rolling(window=window, min_periods=window).var()  # type: ignore[assignment]

    # 0 나눗셈 방지
    x_var_safe: pd.Series = x_var.clip(lower=1e-15)  # type: ignore[assignment]
    beta = xy_cov / x_var_safe

    # Rolling mean
    x_mean = x.rolling(window=window, min_periods=window).mean()
    y_mean = y.rolling(window=window, min_periods=window).mean()

    # alpha = y_mean - beta * x_mean
    alpha = y_mean - beta * x_mean

    # residual = y - (alpha + beta * x)
    residual: pd.Series = y - (alpha + beta * x)  # type: ignore[assignment]
    return residual


def preprocess(df: pd.DataFrame, config: ResidualMomConfig) -> pd.DataFrame:
    """Residual Momentum feature 계산.

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

    # --- Market Factor Proxy ---
    # 자체 수익률의 EWM(장기)을 시장 proxy로 사용
    # 단일 자산 전략이므로 자산 자체의 rolling mean을 시장 trend로 간주
    # 잔차 = 자산 고유 변동 (시장 추세 제거 후)
    market_returns: pd.Series = returns.ewm(  # type: ignore[assignment]
        span=config.regression_window,
        min_periods=config.regression_window,
    ).mean()
    df["market_returns"] = market_returns

    # --- Rolling OLS Residual ---
    residual = _rolling_ols_residual(
        y=returns,
        x=market_returns,
        window=config.regression_window,
    )
    df["residual"] = residual

    # --- Residual Momentum (rolling sum) ---
    df["residual_mom"] = residual.rolling(
        window=config.residual_lookback,
        min_periods=config.residual_lookback,
    ).sum()

    # --- Residual Momentum Z-Score ---
    resid_mom: pd.Series = df["residual_mom"]  # type: ignore[assignment]
    rolling_mean = resid_mom.rolling(
        window=config.zscore_window,
        min_periods=config.zscore_window,
    ).mean()
    rolling_std = resid_mom.rolling(
        window=config.zscore_window,
        min_periods=config.zscore_window,
    ).std()
    df["residual_mom_zscore"] = (resid_mom - rolling_mean) / rolling_std.clip(lower=1e-10)

    # --- Residual Volatility (잔차의 변동성) ---
    df["residual_vol"] = residual.rolling(
        window=config.vol_window,
        min_periods=config.vol_window,
    ).std() * np.sqrt(config.annualization_factor)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
