"""Exchange Flow Momentum 전처리 모듈.

OHLCV + on-chain(oc_flow_net) 데이터에서 거래소 순유출 모멘텀 계산.
oc_flow_net = inflow - outflow (양수 = 유입, 음수 = 유출).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.ex_flow_mom.config import ExFlowMomConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)
from src.market.indicators.composite import rolling_zscore

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "oc_flow_net"})


def preprocess(df: pd.DataFrame, config: ExFlowMomConfig) -> pd.DataFrame:
    """Exchange Flow Momentum feature 계산.

    Args:
        df: OHLCV + oc_flow_net DataFrame
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

    # --- Exchange Net Flow ---
    flow_net: pd.Series = df["oc_flow_net"]  # type: ignore[assignment]
    flow_net = flow_net.ffill()

    # Smoothed net flow (negative = outflow = bullish)
    flow_smooth: pd.Series = flow_net.rolling(  # type: ignore[assignment]
        window=config.flow_window, min_periods=config.flow_window
    ).mean()
    df["flow_smooth"] = flow_smooth

    # Flow momentum: change in flow
    flow_mom: pd.Series = flow_smooth - flow_smooth.shift(config.flow_mom_window)  # type: ignore[assignment]
    df["flow_momentum"] = flow_mom

    # Flow momentum z-score
    df["flow_mom_zscore"] = rolling_zscore(
        flow_mom, window=config.flow_window + config.flow_mom_window
    )

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
