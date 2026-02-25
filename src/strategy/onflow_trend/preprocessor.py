"""OnFlow Trend 전처리 모듈.

8H OHLCV + 1D On-chain(거래소 Flow, MVRV) 데이터에서 feature를 계산.
On-chain 컬럼은 optional (Graceful Degradation).
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.onflow_trend.config import OnflowTrendConfig

from src.market.indicators import (
    drawdown,
    ema,
    log_returns,
    realized_volatility,
    rolling_zscore,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})

# On-chain 컬럼 (optional — Graceful Degradation)
_OC_FLOW_IN = "oc_flow_in_ex_usd"
_OC_FLOW_OUT = "oc_flow_out_ex_usd"
_OC_MVRV = "oc_mvrv"


def preprocess(df: pd.DataFrame, config: OnflowTrendConfig) -> pd.DataFrame:
    """OnFlow Trend feature 계산.

    Calculated columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산, 0~1 scale)
        - vol_scalar: 변동성 스케일러
        - netflow_zscore: 거래소 순입출금 z-score (부재 시 0)
        - mvrv_conviction: MVRV 기반 확신도 (부재 시 1.0 중립)
        - ema_fast / ema_slow: 추세 EMA
        - trend_up: EMA fast > EMA slow (추세 방향)
        - drawdown: rolling drawdown (HEDGE_ONLY용)

    Args:
        df: OHLCV (+ on-chain optional) DataFrame (DatetimeIndex 필수)
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

    # --- Realized Volatility (0~1 scale) ---
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

    # === On-chain Context (Graceful Degradation) ===

    # --- Exchange Netflow z-score ---
    if _OC_FLOW_IN in df.columns and _OC_FLOW_OUT in df.columns:
        flow_in: pd.Series = df[_OC_FLOW_IN].ffill()  # type: ignore[assignment]
        flow_out: pd.Series = df[_OC_FLOW_OUT].ffill()  # type: ignore[assignment]
        net_flow = flow_in - flow_out
        df["netflow_zscore"] = rolling_zscore(net_flow, window=config.flow_zscore_window)
    else:
        df["netflow_zscore"] = 0.0  # 중립

    # --- MVRV Conviction ---
    if _OC_MVRV in df.columns:
        mvrv: pd.Series = df[_OC_MVRV].ffill()  # type: ignore[assignment]
        # MVRV < undervalued → conviction 1.2 (롱 강화)
        # MVRV > overheated → conviction 0.3 (방어)
        # else → 1.0 (중립)
        df["mvrv_conviction"] = pd.Series(
            np.where(
                mvrv < config.mvrv_undervalued,
                1.2,
                np.where(mvrv > config.mvrv_overheated, 0.3, 1.0),
            ),
            index=df.index,
        )
    else:
        df["mvrv_conviction"] = 1.0  # 중립

    # --- Trend Confirmation (EMA Cross) ---
    df["ema_fast"] = ema(close, span=config.trend_ema_fast)
    df["ema_slow"] = ema(close, span=config.trend_ema_slow)
    ema_fast_series: pd.Series = df["ema_fast"]  # type: ignore[assignment]
    ema_slow_series: pd.Series = df["ema_slow"]  # type: ignore[assignment]
    df["trend_up"] = pd.Series(
        np.where(ema_fast_series > ema_slow_series, 1, 0),
        index=df.index,
        dtype=int,
    )

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
