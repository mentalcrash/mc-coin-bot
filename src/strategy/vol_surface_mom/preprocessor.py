"""Volatility Surface Momentum 전처리 모듈.

OHLCV 데이터에서 GK/Parkinson/YZ vol 비율 feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.vol_surface_mom.config import VolSurfaceMomConfig

from src.market.indicators import (
    drawdown,
    garman_klass_volatility,
    log_returns,
    parkinson_volatility,
    realized_volatility,
    roc,
    volatility_scalar,
    yang_zhang_volatility,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: VolSurfaceMomConfig) -> pd.DataFrame:
    """Volatility Surface Momentum feature 계산.

    GK, Parkinson, Yang-Zhang 세 가지 vol 추정치를 계산하고
    이들의 비율(GK/PK, YZ/PK)로 미시구조 정보를 추출한다.

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

    # --- GK Volatility (bar-level → rolling mean) ---
    gk_bar = garman_klass_volatility(open_, high, low, close)
    gk_rolling: pd.Series = gk_bar.rolling(  # type: ignore[assignment]
        window=config.gk_window, min_periods=config.gk_window
    ).mean()
    df["gk_vol"] = gk_rolling

    # --- Parkinson Volatility (bar-level → rolling mean) ---
    pk_bar = parkinson_volatility(high, low)
    pk_rolling: pd.Series = pk_bar.rolling(  # type: ignore[assignment]
        window=config.pk_window, min_periods=config.pk_window
    ).mean()
    df["pk_vol"] = pk_rolling

    # --- Yang-Zhang Volatility (already rolling) ---
    yz_vol = yang_zhang_volatility(open_, high, low, close, window=config.yz_window)
    df["yz_vol"] = yz_vol

    # --- Vol Ratios ---
    # GK/PK ratio: >1 = close-to-close 변동 우세, <1 = range 변동 우세
    pk_safe = pk_rolling.clip(lower=1e-10)
    gk_pk_ratio: pd.Series = gk_rolling / pk_safe  # type: ignore[assignment]
    df["gk_pk_ratio"] = gk_pk_ratio.rolling(
        window=config.ratio_window, min_periods=config.ratio_window
    ).mean()

    # YZ/PK ratio: overnight + intraday 구조 변화 탐지
    yz_safe = yz_vol.clip(lower=1e-10)
    yz_pk_ratio: pd.Series = yz_safe / pk_safe  # type: ignore[assignment]
    df["yz_pk_ratio"] = yz_pk_ratio.rolling(
        window=config.ratio_window, min_periods=config.ratio_window
    ).mean()

    # --- Momentum (ROC) ---
    df["momentum"] = roc(close, period=config.momentum_window)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
