"""VRP-Regime Trend 전처리 모듈.

OHLCV + DVOL(Options) 데이터에서 VRP(Volatility Risk Premium) + 레짐 feature를 계산.
핵심: Garman-Klass RV vs DVOL(IV) 스프레드 + EMA cross 추세 확인.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.vrp_regime_trend.config import VrpRegimeTrendConfig

from src.market.indicators import (
    drawdown,
    ema,
    garman_klass_volatility,
    log_returns,
    realized_volatility,
    rolling_zscore,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})

# DVOL (Options) — optional for Graceful Degradation
_OPT_DVOL = "opt_dvol"
# Deribit DVOL columns injected by OptionsDataService (asset-specific)
_DVOL_CANDIDATES = ("opt_dvol", "opt_btc_dvol", "opt_eth_dvol")


def _resolve_dvol_column(df: pd.DataFrame) -> str | None:
    """DataFrame에서 사용 가능한 DVOL 컬럼을 찾는다.

    우선순위: opt_dvol > opt_btc_dvol > opt_eth_dvol.
    BTC 전용 전략이므로 opt_btc_dvol 우선.

    Returns:
        컬럼명 또는 None (DVOL 부재)
    """
    for col in _DVOL_CANDIDATES:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def preprocess(df: pd.DataFrame, config: VrpRegimeTrendConfig) -> pd.DataFrame:
    """VRP-Regime Trend feature 계산.

    Calculated columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산, 0~1 scale)
        - vol_scalar: 변동성 스케일러
        - gk_rv: Garman-Klass Realized Volatility (bar 단위)
        - gk_rv_ann_pct: GK RV 연환산 % (DVOL 단위 일치)
        - dvol_clean: DVOL forward-filled (annualized %)
        - vrp: Volatility Risk Premium = DVOL - GK_RV_ann (%)
        - vrp_ma: VRP 이동평균
        - vrp_zscore: VRP z-score
        - ema_fast / ema_slow: 추세 EMA
        - trend_up: EMA fast > EMA slow (추세 방향)
        - drawdown: rolling drawdown (HEDGE_ONLY용)

    Args:
        df: OHLCV (+ opt_dvol optional) DataFrame (DatetimeIndex 필수)
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

    # --- Realized Volatility (0~1 scale, for vol_scalar) ---
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

    # --- Garman-Klass RV (bar-level volatility estimator) ---
    gk_rv = garman_klass_volatility(open_, high, low, close)
    # Rolling mean for smoothing + annualization to % (DVOL 단위 일치)
    gk_rv_rolling: pd.Series = gk_rv.rolling(  # type: ignore[assignment]
        window=config.gk_rv_window, min_periods=config.gk_rv_window
    ).mean()
    df["gk_rv"] = gk_rv_rolling
    # Annualize: bar-level variance → annual std → percentage
    gk_rv_ann_pct: pd.Series = (  # type: ignore[assignment]
        np.sqrt(gk_rv_rolling.clip(lower=0) * config.annualization_factor) * 100.0
    )
    df["gk_rv_ann_pct"] = gk_rv_ann_pct

    # --- DVOL (Implied Volatility) — Graceful Degradation ---
    dvol_col = _resolve_dvol_column(df)
    if dvol_col is not None:
        dvol: pd.Series = df[dvol_col]  # type: ignore[assignment]
        dvol_clean = dvol.ffill()
        df["dvol_clean"] = dvol_clean
    else:
        # DVOL 부재 시 GK RV 기반 추정 (VRP = 0에 수렴 → 중립)
        df["dvol_clean"] = gk_rv_ann_pct

    # --- VRP (Volatility Risk Premium) ---
    # VRP > 0: IV 과대평가 (공포 프리미엄) → Long 우위
    # VRP < 0: IV 과소평가 (실제 리스크) → Short 우위
    dvol_series: pd.Series = df["dvol_clean"]  # type: ignore[assignment]
    vrp: pd.Series = dvol_series - gk_rv_ann_pct  # type: ignore[assignment]
    df["vrp"] = vrp

    # --- VRP Moving Average (smoothing) ---
    vrp_ma: pd.Series = vrp.rolling(  # type: ignore[assignment]
        window=config.vrp_ma_window, min_periods=config.vrp_ma_window
    ).mean()
    df["vrp_ma"] = vrp_ma

    # --- VRP Z-Score ---
    vrp_z = rolling_zscore(vrp_ma, window=config.vrp_zscore_window)
    # Graceful Degradation: DVOL 부재 시 std=0 → NaN → 0 (중립)
    df["vrp_zscore"] = vrp_z.fillna(0.0)

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
