"""Volatility Structure ML 전처리 모듈.

OHLCV 데이터에서 13종 vol 기반 feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.vol_struct_ml.config import VolStructMLConfig

from src.market.indicators import (
    adx,
    atr,
    drawdown,
    efficiency_ratio,
    fractal_dimension,
    garman_klass_volatility,
    hurst_exponent,
    log_returns,
    parkinson_volatility,
    realized_volatility,
    vol_percentile_rank,
    volatility_of_volatility,
    volatility_scalar,
    yang_zhang_volatility,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: VolStructMLConfig) -> pd.DataFrame:
    """Volatility Structure ML feature 계산.

    13종 vol-based features + returns/vol_scalar/forward_return.

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
    high: pd.Series = df["high"]  # type: ignore[assignment]
    low: pd.Series = df["low"]  # type: ignore[assignment]
    open_: pd.Series = df["open"]  # type: ignore[assignment]

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

    # === 13 Vol-Based Features ===
    w = config.vol_estimator_window

    # 1. Garman-Klass Volatility
    gk_vol = garman_klass_volatility(open_, high, low, close)
    df["feat_gk_vol"] = gk_vol.rolling(w, min_periods=w).mean()

    # 2. Parkinson Volatility (rolling)
    park_raw = parkinson_volatility(high, low)
    df["feat_park_vol"] = park_raw.rolling(w, min_periods=w).mean()

    # 3. Yang-Zhang Volatility
    df["feat_yz_vol"] = yang_zhang_volatility(open_, high, low, close, window=w)

    # 4. VoV (Volatility of Volatility)
    df["feat_vov"] = volatility_of_volatility(realized_vol, window=config.vov_window)

    # 5. Vol Percentile Rank
    df["feat_vol_pctrank"] = vol_percentile_rank(realized_vol, window=config.hurst_window)

    # 6. Fractal Dimension
    df["feat_fractal_dim"] = fractal_dimension(close, period=config.fractal_period)

    # 7. Hurst Exponent
    df["feat_hurst"] = hurst_exponent(close, window=config.hurst_window)

    # 8. Efficiency Ratio
    df["feat_er"] = efficiency_ratio(close, period=config.er_period)

    # 9. ADX
    df["feat_adx"] = adx(high, low, close, period=config.adx_period)

    # 10. ATR Ratio (normalized by close)
    atr_val = atr(high, low, close, period=config.adx_period)
    close_safe = close.replace(0, np.nan)
    df["feat_atr_ratio"] = atr_val / close_safe

    # 11. Short/Long Vol Ratio
    short_vol = returns.rolling(10, min_periods=10).std()
    long_vol = returns.rolling(config.vol_window, min_periods=config.vol_window).std()
    df["feat_vol_ratio"] = short_vol / long_vol.clip(lower=1e-10)

    # 12. Vol Regime (rolling zscore of vol)
    vol_mean = realized_vol.rolling(config.hurst_window, min_periods=config.hurst_window).mean()
    vol_std = realized_vol.rolling(config.hurst_window, min_periods=config.hurst_window).std()
    df["feat_vol_zscore"] = (realized_vol - vol_mean) / vol_std.clip(lower=1e-10)

    # 13. Returns Skewness (rolling)
    df["feat_ret_skew"] = returns.rolling(config.vol_window, min_periods=config.vol_window).skew()

    # --- Forward Return (training target) ---
    df["forward_return"] = close.pct_change(config.prediction_horizon).shift(
        -config.prediction_horizon
    )

    # --- Drawdown (HEDGE_ONLY) ---
    df["drawdown"] = drawdown(close)

    # --- ATR (trailing stop) ---
    df["atr"] = atr_val

    return df
