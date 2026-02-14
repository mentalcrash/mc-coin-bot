"""Variance Decomposition Momentum 전처리 모듈.

OHLCV 데이터에서 good/bad semivariance 분해 feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.vardecomp_mom.config import VardecompMomConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    rolling_return,
    volatility_scalar,
)


_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: VardecompMomConfig) -> pd.DataFrame:
    """Variance Decomposition Momentum feature 계산.

    Calculated columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - good_semivar: 상방 수익률의 rolling semivariance
        - bad_semivar: 하방 수익률의 rolling semivariance
        - var_ratio: good_semivar / (good_semivar + bad_semivar)
        - price_mom: 가격 모멘텀 (rolling return)
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

    # --- Good/Bad Semivariance ---
    # Good semivariance: 상방 수익률만의 variance (positive returns)
    positive_returns_sq = np.where(returns > 0, returns**2, 0.0)
    good_sv = pd.Series(positive_returns_sq, index=df.index).rolling(
        window=config.semivar_window, min_periods=config.semivar_window
    ).mean()
    df["good_semivar"] = good_sv

    # Bad semivariance: 하방 수익률만의 variance (negative returns)
    negative_returns_sq = np.where(returns < 0, returns**2, 0.0)
    bad_sv = pd.Series(negative_returns_sq, index=df.index).rolling(
        window=config.semivar_window, min_periods=config.semivar_window
    ).mean()
    df["bad_semivar"] = bad_sv

    # --- Variance Ratio ---
    # var_ratio = good_semivar / (good_semivar + bad_semivar)
    # > 0.5 이면 상방 변동성 지배적 (건강한 추세)
    # < 0.5 이면 하방 변동성 지배적 (취약한 추세)
    total_sv = good_sv + bad_sv
    df["var_ratio"] = good_sv / total_sv.clip(lower=1e-20)

    # --- Price Momentum ---
    df["price_mom"] = rolling_return(close, window=config.mom_lookback)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
