"""ML Derivatives Regime 전처리 모듈 (Derivatives).

OHLCV + funding_rate 데이터에서 derivatives-only ML feature를 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.ml_deriv_regime.config import MlDerivRegimeConfig

from src.market.indicators import (
    drawdown,
    funding_rate_ma,
    funding_zscore,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "funding_rate"})


def _compute_deriv_features(
    df: pd.DataFrame,
    config: MlDerivRegimeConfig,
) -> pd.DataFrame:
    """Derivatives-only ML features 계산.

    Returns:
        feat_ 접두사의 feature DataFrame
    """
    funding_rate: pd.Series = df["funding_rate"]  # type: ignore[assignment]
    funding_rate = funding_rate.ffill()  # merge_asof 후 NaN 처리
    close: pd.Series = df["close"]  # type: ignore[assignment]

    features: dict[str, pd.Series] = {}

    # --- Funding Rate Features ---
    # 1. Short-term FR MA
    features["feat_fr_ma_short"] = funding_rate_ma(funding_rate, window=config.fr_lookback_short)

    # 2. Long-term FR MA
    features["feat_fr_ma_long"] = funding_rate_ma(funding_rate, window=config.fr_lookback_long)

    # 3. FR Z-score (short window)
    features["feat_fr_zscore_short"] = funding_zscore(
        funding_rate, ma_window=config.fr_lookback_short, zscore_window=config.fr_zscore_window
    )

    # 4. FR Z-score (long window)
    features["feat_fr_zscore_long"] = funding_zscore(
        funding_rate, ma_window=config.fr_lookback_long, zscore_window=config.fr_zscore_window
    )

    # 5. FR momentum (short - long)
    fr_short = features["feat_fr_ma_short"]
    fr_long = features["feat_fr_ma_long"]
    features["feat_fr_momentum"] = fr_short - fr_long

    # 6. FR acceleration (change in FR)
    features["feat_fr_accel"] = funding_rate.diff()

    # 7. FR rolling std (volatility of FR)
    fr_vol: pd.Series = funding_rate.rolling(config.fr_lookback_long).std()  # type: ignore[assignment]
    features["feat_fr_vol"] = fr_vol

    # 8. FR cumulative sum (directional pressure)
    fr_cumsum: pd.Series = funding_rate.rolling(config.fr_lookback_long).sum()  # type: ignore[assignment]
    features["feat_fr_cumsum"] = fr_cumsum

    # 9. FR sign persistence (how many bars same sign)
    fr_sign: pd.Series = np.sign(funding_rate)  # type: ignore[assignment]
    sign_change = fr_sign != fr_sign.shift(1)
    sign_group = sign_change.cumsum()
    features["feat_fr_sign_persist"] = sign_group.groupby(sign_group).cumcount().astype(float)

    # 10. Price-FR divergence: returns vs FR direction
    returns_5d = np.log(close / close.shift(5))
    fr_dir = np.sign(features["feat_fr_ma_short"])
    ret_dir = np.sign(returns_5d)
    features["feat_price_fr_div"] = (ret_dir * fr_dir).astype(float)

    return pd.DataFrame(features, index=df.index)


def preprocess(df: pd.DataFrame, config: MlDerivRegimeConfig) -> pd.DataFrame:
    """ML Derivatives Regime feature 계산 (Derivatives 포함).

    Args:
        df: OHLCV + funding_rate DataFrame (DatetimeIndex 필수)
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

    # --- Derivatives Features ---
    feat_df = _compute_deriv_features(df, config)
    for col in feat_df.columns:
        df[col] = feat_df[col]

    # --- Regime Features (if available, from RegimeService) ---
    # regime 컬럼이 있으면 feat_ 접두사로 복사 (ML input)
    if "p_trending" in df.columns:
        df["feat_p_trending"] = df["p_trending"]
        df["feat_p_ranging"] = df["p_ranging"]
        df["feat_p_volatile"] = df["p_volatile"]

    # --- Forward Return (ML training target) ---
    df["forward_return"] = close.pct_change(config.prediction_horizon).shift(
        -config.prediction_horizon
    )

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
