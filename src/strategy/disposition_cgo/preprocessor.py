"""Disposition CGO 전처리 모듈.

OHLCV 데이터에서 turnover-weighted reference price, CGO, overhang spread feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.disposition_cgo.config import DispositionCgoConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    rolling_return,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: DispositionCgoConfig) -> pd.DataFrame:
    """Disposition CGO feature 계산.

    Calculated columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - reference_price: turnover(volume) 가중 EWM 평균 매입단가 추정
        - cgo: Capital Gains Overhang = (price - ref_price) / ref_price
        - cgo_smooth: CGO의 EMA smoothing
        - momentum: rolling simple return (momentum_window)
        - overhang_spread: CGO * sign(momentum) — disposition-momentum alignment
        - overhang_spread_ma: overhang_spread의 rolling mean
        - drawdown: rolling drawdown (HEDGE_ONLY용)

    Reference Price 계산 (Grinblatt & Han 2005):
        volume을 proxy turnover로 사용, EWM(span=turnover_window) 가중.
        ref_price = ewm(volume * close) / ewm(volume)

    Overhang Spread (Frazzini 2006):
        CGO와 momentum의 상호작용을 측정.
        높은 CGO + 양의 momentum → disposition effect로 인한 underreaction drift.

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
    volume: pd.Series = df["volume"]  # type: ignore[assignment]

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

    # --- Reference Price (Turnover-Weighted Average Cost) ---
    # Grinblatt & Han (2005): volume을 proxy turnover로 사용
    vol_x_price = volume * close
    ewm_vol_price: pd.Series = vol_x_price.ewm(  # type: ignore[assignment]
        span=config.turnover_window, min_periods=config.turnover_window
    ).mean()
    ewm_vol: pd.Series = volume.ewm(  # type: ignore[assignment]
        span=config.turnover_window, min_periods=config.turnover_window
    ).mean()
    ref_price = ewm_vol_price / ewm_vol.clip(lower=1e-10)
    df["reference_price"] = ref_price

    # --- Capital Gains Overhang ---
    # CGO = (current_price - reference_price) / reference_price
    cgo = (close - ref_price) / ref_price.clip(lower=1e-10)
    df["cgo"] = cgo

    # --- CGO Smoothing (EMA) ---
    cgo_smooth: pd.Series = cgo.ewm(  # type: ignore[assignment]
        span=config.cgo_smooth_window, min_periods=config.cgo_smooth_window
    ).mean()
    df["cgo_smooth"] = cgo_smooth

    # --- Momentum (simple rolling return) ---
    momentum = rolling_return(close, period=config.momentum_window)
    df["momentum"] = momentum

    # --- Overhang Spread (Frazzini 2006) ---
    # CGO * sign(momentum) — 양수면 CGO와 momentum 방향 일치
    mom_sign = np.sign(momentum)
    overhang_spread: pd.Series = cgo_smooth * mom_sign  # type: ignore[assignment]
    df["overhang_spread"] = overhang_spread

    # --- Overhang Spread Rolling Mean ---
    spread_ma: pd.Series = overhang_spread.rolling(  # type: ignore[assignment]
        window=config.overhang_spread_window,
        min_periods=config.overhang_spread_window,
    ).mean()
    df["overhang_spread_ma"] = spread_ma

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
