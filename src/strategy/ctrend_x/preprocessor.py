"""CTREND-X 전처리 모듈.

CTREND 동일 28개 feature + returns/vol_scalar/forward_return 계산.
compute_all_features()는 CTREND 모듈에서 재사용.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.market.indicators import (
    log_returns,
    realized_volatility,
    volatility_scalar,
)
from src.strategy.ctrend.preprocessor import compute_all_features

if TYPE_CHECKING:
    from src.strategy.ctrend_x.config import CTRENDXConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: CTRENDXConfig) -> pd.DataFrame:
    """CTREND-X feature 계산 (28 features + vol_scalar + forward_return).

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수).
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

    # OHLCV를 float64로 변환
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 28개 feature 계산 (CTREND 모듈 재사용)
    feature_df = compute_all_features(result)
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

    # Forward return (training target)
    result["forward_return"] = close.pct_change(config.prediction_horizon).shift(
        -config.prediction_horizon
    )

    return result
