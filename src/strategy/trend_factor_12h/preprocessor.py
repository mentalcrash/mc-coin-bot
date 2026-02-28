"""Trend Factor Multi-Horizon 전처리 모듈.

5-horizon risk-adjusted return(ret_h/vol_h) feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    from src.strategy.trend_factor_12h.config import TrendFactorConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def _rolling_risk_adjusted_return(returns: pd.Series, horizon: int) -> pd.Series:
    """Rolling risk-adjusted return: cum_ret_h / vol_h.

    각 horizon에서 수익률 합산(log return sum = log cum return)을
    해당 구간 변동성(std * sqrt(horizon))으로 나눈다.

    Args:
        returns: log return series
        horizon: lookback window (bars)

    Returns:
        Rolling risk-adjusted return (trend factor component)
    """
    # Cumulative return over horizon (log space: sum of log returns)
    cum_ret = returns.rolling(window=horizon, min_periods=horizon).sum()
    # Volatility over horizon: std * sqrt(horizon)
    vol_h = returns.rolling(window=horizon, min_periods=horizon).std() * np.sqrt(horizon)
    # 0 나눗셈 방어
    return cum_ret / vol_h.clip(lower=1e-10)  # type: ignore[return-value]


def preprocess(df: pd.DataFrame, config: TrendFactorConfig) -> pd.DataFrame:
    """Trend Factor Multi-Horizon feature 계산.

    Calculated Columns:
        - returns: log return
        - realized_vol: 연환산 실현 변동성
        - vol_scalar: 변동성 스케일러
        - tf_h{N}: 각 horizon의 risk-adjusted return
        - trend_factor: 5-horizon risk-adjusted return 합산
        - drawdown: HEDGE_ONLY용 drawdown

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

    # --- 5-Horizon Risk-Adjusted Returns ---
    horizons = [
        config.horizon_1,
        config.horizon_2,
        config.horizon_3,
        config.horizon_4,
        config.horizon_5,
    ]

    tf_components: list[pd.Series] = []
    for h in horizons:
        tf_h = _rolling_risk_adjusted_return(returns, h)
        df[f"tf_h{h}"] = tf_h
        tf_components.append(tf_h)

    # --- Trend Factor: 5-horizon 합산 ---
    trend_factor = pd.concat(tf_components, axis=1).sum(axis=1)
    df["trend_factor"] = trend_factor

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
