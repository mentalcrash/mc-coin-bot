"""CVD Divergence 8H 전처리 모듈.

OHLCV + optional dext_cvd_buy_vol (Coinalyze daily CVD buy_volume, merge_asof forward-fill).
CVD 컬럼 없으면 graceful degradation → EMA trend only.

Note: Coinalyze CVD는 daily 데이터. 8H (3 bars/day)에서 동일 daily 값이
forward-fill되어 3회 반복된다. CVD는 방향적 context이므로 이는 의도된 동작.
"""

import numpy as np
import pandas as pd

from src.market.indicators import (
    drawdown,
    ema,
    log_returns,
    realized_volatility,
    volatility_scalar,
)
from src.strategy.cvd_diverge_8h.config import CvdDiverge8hConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})

# Coinalyze CVD buy_volume column (injected by deriv_ext service)
_CVD_COLUMN = "dext_cvd_buy_vol"


def preprocess(df: pd.DataFrame, config: CvdDiverge8hConfig) -> pd.DataFrame:
    """CVD Divergence 8H feature 계산.

    Calculated columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - trend_ema: 추세 확인 EMA
        - has_cvd: CVD 데이터 존재 여부 (bool scalar, 컬럼 아님)
        - cvd_smooth: CVD EMA (CVD 존재 시)
        - price_smooth: 가격 EMA
        - cvd_roc: CVD ROC (CVD 존재 시)
        - price_roc: 가격 ROC
        - divergence_zscore: Divergence z-score (CVD 존재 시, 아니면 0)
        - drawdown: rolling drawdown (HEDGE_ONLY용)

    Args:
        df: OHLCV + optional dext_cvd_buy_vol DataFrame
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

    # --- Trend EMA ---
    df["trend_ema"] = ema(close, span=config.trend_ema_window)

    # --- Price smoothing + ROC ---
    price_smooth = ema(close, span=config.price_ma_window)
    df["price_smooth"] = price_smooth
    price_smooth_shifted = price_smooth.shift(config.cvd_lookback)
    df["price_roc"] = (price_smooth - price_smooth_shifted) / price_smooth_shifted.clip(lower=1e-10)

    # --- CVD features (graceful degradation) ---
    has_cvd = _CVD_COLUMN in df.columns and bool(df[_CVD_COLUMN].notna().any())

    if has_cvd:
        cvd_raw: pd.Series = df[_CVD_COLUMN].ffill()  # type: ignore[assignment]
        cvd_smooth = ema(cvd_raw, span=config.cvd_ma_window)
        df["cvd_smooth"] = cvd_smooth

        cvd_smooth_shifted = cvd_smooth.shift(config.cvd_lookback)
        cvd_roc: pd.Series = (cvd_smooth - cvd_smooth_shifted) / cvd_smooth_shifted.abs().clip(
            lower=1e-10
        )  # type: ignore[assignment]
        df["cvd_roc"] = cvd_roc

        # Divergence score: price_roc - cvd_roc
        # Positive = price rising faster than CVD (bearish divergence)
        # Negative = CVD rising faster than price (bullish divergence)
        price_roc: pd.Series = df["price_roc"]  # type: ignore[assignment]
        divergence_raw = price_roc - cvd_roc

        # Normalize to z-score
        div_mean: pd.Series = divergence_raw.rolling(  # type: ignore[assignment]
            window=config.cvd_lookback, min_periods=config.cvd_lookback
        ).mean()
        div_std: pd.Series = divergence_raw.rolling(  # type: ignore[assignment]
            window=config.cvd_lookback, min_periods=config.cvd_lookback
        ).std()
        df["divergence_zscore"] = (divergence_raw - div_mean) / div_std.clip(lower=1e-10)
    else:
        # No CVD data → neutral divergence (pure EMA trend fallback)
        df["cvd_smooth"] = np.nan
        df["cvd_roc"] = 0.0
        df["divergence_zscore"] = 0.0

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
