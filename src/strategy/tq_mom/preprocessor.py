"""Trend Quality Momentum (TQ-Mom) 전처리 모듈.

OHLCV 데이터에서 Hurst exponent, Fractal Dimension, 모멘텀 feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd  # noqa: TC002

if TYPE_CHECKING:
    from src.strategy.tq_mom.config import TqMomConfig

from src.market.indicators import (
    drawdown,
    fractal_dimension,
    hurst_exponent,
    log_returns,
    realized_volatility,
    rolling_return,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: TqMomConfig) -> pd.DataFrame:
    """Trend Quality Momentum feature 계산.

    Calculated columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - hurst: Hurst exponent (0~1; > 0.5 = 추세 지속)
        - fd: Fractal Dimension (1~2; ≈ 1 = 추세, ≈ 1.5 = 랜덤)
        - price_mom: 가격 모멘텀
        - drawdown: rolling drawdown (HEDGE_ONLY용)

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
    df["hurst"] = hurst_exponent(close, window=config.hurst_window)

    # --- Fractal Dimension ---
    df["fd"] = fractal_dimension(close, period=config.fd_period)

    # --- Price Momentum ---
    df["price_mom"] = rolling_return(close, period=config.mom_lookback)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
