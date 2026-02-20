"""VRP-Trend 전처리 모듈.

OHLCV + DVOL(Options) 데이터에서 VRP(Volatility Risk Premium) feature를 계산한다.
VRP = IV(DVOL) - RV(Realized Vol). 모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.vrp_trend.config import VrpTrendConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    rolling_zscore,
    sma,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "dvol"})


def preprocess(df: pd.DataFrame, config: VrpTrendConfig) -> pd.DataFrame:
    """VRP-Trend feature 계산.

    Calculated columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산, %)
        - vol_scalar: 변동성 스케일러
        - dvol_clean: DVOL forward-filled (annualized %)
        - rv_annualized_pct: RV를 DVOL 단위(%)로 변환
        - vrp: Volatility Risk Premium = DVOL - RV (%)
        - vrp_ma: VRP 이동평균
        - vrp_zscore: VRP z-score
        - trend_sma: 추세 확인 SMA
        - above_trend: close > SMA (추세 방향)
        - drawdown: rolling drawdown (HEDGE_ONLY용)

    Args:
        df: OHLCV + dvol DataFrame (DatetimeIndex 필수)
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

    # --- DVOL (Implied Volatility, annualized %) ---
    dvol: pd.Series = df["dvol"]  # type: ignore[assignment]
    dvol_clean = dvol.ffill()  # merge_asof 후 NaN 처리
    df["dvol_clean"] = dvol_clean

    # --- RV for VRP calculation (annualized %, DVOL 단위와 일치) ---
    rv_for_vrp = realized_volatility(
        returns,
        window=config.rv_window,
        annualization_factor=config.annualization_factor,
    )
    # DVOL은 annualized % (예: 55.0 = 55%), RV도 동일 단위로 변환
    rv_annualized_pct: pd.Series = rv_for_vrp * 100.0  # type: ignore[assignment]
    df["rv_annualized_pct"] = rv_annualized_pct

    # --- VRP (Volatility Risk Premium) ---
    # VRP > 0: IV 과대평가 (공포 프리미엄) → 시장 과공포 → Long 우위
    # VRP < 0: IV 과소평가 (실제 리스크 높음) → Short 우위
    vrp: pd.Series = dvol_clean - rv_annualized_pct  # type: ignore[assignment]
    df["vrp"] = vrp

    # --- VRP Moving Average (smoothing) ---
    vrp_ma: pd.Series = vrp.rolling(  # type: ignore[assignment]
        window=config.vrp_ma_window, min_periods=config.vrp_ma_window
    ).mean()
    df["vrp_ma"] = vrp_ma

    # --- VRP Z-Score ---
    df["vrp_zscore"] = rolling_zscore(vrp_ma, window=config.vrp_zscore_window)

    # --- Trend Confirmation (SMA) ---
    df["trend_sma"] = sma(close, period=config.trend_sma_window)
    trend_sma_series: pd.Series = df["trend_sma"]  # type: ignore[assignment]
    df["above_trend"] = pd.Series(
        np.where(close > trend_sma_series, 1, 0),
        index=df.index,
        dtype=int,
    )

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
