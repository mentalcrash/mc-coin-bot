"""MVRV Cycle Trend 전처리 모듈.

OHLCV + On-chain(MVRV) 데이터에서 전략 feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).

Features:
    - returns: log returns
    - realized_vol / vol_scalar: volatility targeting
    - mom_fast / mom_slow / mom_blend: multi-lookback momentum
    - mvrv_zscore: MVRV Z-Score (사이클 레짐 필터)
    - drawdown: HEDGE_ONLY drawdown
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.mvrv_cycle_trend.config import MvrvCycleTrendConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    mvrv_zscore,
    realized_volatility,
    rolling_return,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})

# On-chain 컬럼 (optional — Graceful Degradation)
_OC_MVRV = "oc_mvrv"


def preprocess(df: pd.DataFrame, config: MvrvCycleTrendConfig) -> pd.DataFrame:
    """MVRV Cycle Trend feature 계산.

    Args:
        df: OHLCV + optional on-chain DataFrame (DatetimeIndex 필수)
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

    # --- Multi-Lookback Momentum (12H bars) ---
    df["mom_fast"] = rolling_return(close, period=config.mom_fast)
    df["mom_slow"] = rolling_return(close, period=config.mom_slow)

    # Blended momentum: weighted average of fast and slow
    mom_fast_series: pd.Series = df["mom_fast"]  # type: ignore[assignment]
    mom_slow_series: pd.Series = df["mom_slow"]  # type: ignore[assignment]
    df["mom_blend"] = (
        config.mom_blend_weight * mom_fast_series
        + (1.0 - config.mom_blend_weight) * mom_slow_series
    )

    # --- MVRV Z-Score (On-chain, Graceful Degradation) ---
    if _OC_MVRV in df.columns:
        mvrv_raw: pd.Series = df[_OC_MVRV].ffill()  # type: ignore[assignment]
        df["mvrv_zscore"] = mvrv_zscore(mvrv_raw, window=config.mvrv_zscore_window)
    else:
        # Graceful Degradation: NaN → signal.py에서 중립(0) 처리
        df["mvrv_zscore"] = np.nan

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
