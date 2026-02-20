"""Acceleration-Volatility Trend 전처리 모듈.

OHLCV 데이터에서 가격 가속도 + GK vol 정규화 feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.accel_vol_trend.config import AccelVolTrendConfig

from src.market.indicators import (
    drawdown,
    garman_klass_volatility,
    log_returns,
    price_acceleration,
    realized_volatility,
    roc,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: AccelVolTrendConfig) -> pd.DataFrame:
    """Acceleration-Volatility Trend feature 계산.

    가격 가속도(fast ROC - slow ROC)와 GK volatility를 계산하여
    vol-normalized 가속도를 생성한다.

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
    open_: pd.Series = df["open"]  # type: ignore[assignment]

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

    # --- Price Acceleration (2nd derivative) ---
    raw_accel = price_acceleration(close, fast=config.accel_fast, slow=config.accel_slow)
    # Smoothing to reduce noise
    df["acceleration"] = raw_accel.rolling(
        window=config.accel_smooth, min_periods=config.accel_smooth
    ).mean()

    # --- GK Volatility (bar-level -> rolling) ---
    gk_bar = garman_klass_volatility(open_, high, low, close)
    gk_rolling: pd.Series = gk_bar.rolling(  # type: ignore[assignment]
        window=config.gk_window, min_periods=config.gk_window
    ).mean()
    df["gk_vol"] = gk_rolling

    # --- Vol-Normalized Acceleration ---
    # 가속도를 GK vol로 정규화하여 vol regime에 무관한 시그널 생성
    gk_safe = gk_rolling.clip(lower=1e-10)
    df["norm_acceleration"] = df["acceleration"] / gk_safe

    # --- Momentum (ROC) ---
    df["momentum"] = roc(close, period=config.momentum_window)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
