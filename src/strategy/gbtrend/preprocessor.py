"""GBTrend 전처리 모듈.

모멘텀 중심 12개 feature 계산. CTREND 대비 축소된 feature set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.market.indicators import (
    adx,
    atr,
    ema_cross,
    log_returns,
    momentum,
    realized_volatility,
    roc,
    rsi,
    volatility_scalar,
)

if TYPE_CHECKING:
    from src.strategy.gbtrend.config import GBTrendConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def _compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """12개 모멘텀 중심 feature 계산.

    Args:
        df: OHLCV DataFrame.

    Returns:
        12개 feat_ 컬럼을 가진 DataFrame.
    """
    close: pd.Series = df["close"]  # type: ignore[assignment]
    high: pd.Series = df["high"]  # type: ignore[assignment]
    low: pd.Series = df["low"]  # type: ignore[assignment]

    features: dict[str, pd.Series] = {}

    # 1-4. ROC 4종 (multi-horizon)
    features["feat_roc_5"] = roc(close, period=5)
    features["feat_roc_10"] = roc(close, period=10)
    features["feat_roc_21"] = roc(close, period=21)
    features["feat_roc_63"] = roc(close, period=63)

    # 5-6. EMA Cross 2종
    features["feat_ema_cross_5_20"] = ema_cross(close, fast=5, slow=20)
    features["feat_ema_cross_10_50"] = ema_cross(close, fast=10, slow=50)

    # 7. RSI (14)
    features["feat_rsi_14"] = rsi(close, period=14)

    # 8. ADX (14) — trend strength
    features["feat_adx_14"] = adx(high, low, close, period=14)

    # 9. ATR Ratio (14) — normalized volatility
    atr_14 = atr(high, low, close, period=14)
    close_safe = close.replace(0, np.nan)
    features["feat_atr_ratio_14"] = atr_14 / close_safe

    # 10. Vol Ratio (short/long)
    short_vol = close.pct_change().rolling(10).std()
    long_vol = close.pct_change().rolling(30).std().replace(0, np.nan)
    features["feat_vol_ratio"] = short_vol / long_vol

    # 11-12. Momentum (normalized by close)
    mom_5 = momentum(close, period=5)
    mom_21 = momentum(close, period=21)
    features["feat_mom_5"] = mom_5 / close_safe
    features["feat_mom_21"] = mom_21 / close_safe

    return pd.DataFrame(features, index=df.index)


def preprocess(df: pd.DataFrame, config: GBTrendConfig) -> pd.DataFrame:
    """GBTrend feature 계산.

    Args:
        df: OHLCV DataFrame.
        config: 전략 설정.

    Returns:
        feature가 추가된 새 DataFrame.

    Raises:
        ValueError: 필수 컬럼 누락 또는 빈 DataFrame.
    """
    if df.empty:
        msg = "Input DataFrame is empty"
        raise ValueError(msg)

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    result = df.copy()

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 12개 feature
    feature_df = _compute_momentum_features(result)
    for col in feature_df.columns:
        result[col] = feature_df[col]

    close: pd.Series = result["close"]  # type: ignore[assignment]

    # Returns
    returns = log_returns(close)
    result["returns"] = returns

    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # Realized Volatility
    result["realized_vol"] = realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # Vol Scalar
    result["vol_scalar"] = volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # Forward return
    result["forward_return"] = close.pct_change(config.prediction_horizon).shift(
        -config.prediction_horizon
    )

    return result
