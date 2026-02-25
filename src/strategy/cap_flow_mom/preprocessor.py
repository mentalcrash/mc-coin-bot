"""Capital Flow Momentum 전처리 모듈.

OHLCV(12H) + On-chain Stablecoin supply(1D) feature 계산.
모든 연산은 벡터화 (for 루프 금지).

Features:
    - fast_roc: 빠른 ROC (가격 모멘텀 단기)
    - slow_roc: 느린 ROC (가격 모멘텀 장기)
    - stablecoin_roc: Stablecoin total supply ROC (자본 흐름)
    - realized_vol / vol_scalar: Volatility targeting
    - drawdown: HEDGE_ONLY용
"""

import numpy as np
import pandas as pd

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    roc,
    volatility_scalar,
)
from src.strategy.cap_flow_mom.config import CapFlowMomConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})

# On-chain columns (optional — Graceful Degradation)
_OC_STABLECOIN = "oc_stablecoin_total_usd"


def preprocess(df: pd.DataFrame, config: CapFlowMomConfig) -> pd.DataFrame:
    """Capital Flow Momentum feature 계산.

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

    # --- Dual-Speed ROC (price momentum) ---
    df["fast_roc"] = roc(close, period=config.fast_roc_period)
    df["slow_roc"] = roc(close, period=config.slow_roc_period)

    # --- Stablecoin Supply ROC (capital flow context, Graceful Degradation) ---
    if _OC_STABLECOIN in df.columns:
        stab_series: pd.Series = df[_OC_STABLECOIN].ffill()  # type: ignore[assignment]
        df["stablecoin_roc"] = stab_series.pct_change(config.stablecoin_roc_window)
    else:
        df["stablecoin_roc"] = np.nan  # NaN → signal에서 중립(1.0) 처리

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
