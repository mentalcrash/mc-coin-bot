"""Vol-Term ML 전처리 모듈.

다중 RV term structure feature 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.market.indicators import (
    garman_klass_volatility,
    log_returns,
    parkinson_volatility,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    from src.strategy.vol_term_ml.config import VolTermMLConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})

# RV 계산 윈도우 목록
_RV_WINDOWS = (5, 10, 20, 40, 60)


def preprocess(df: pd.DataFrame, config: VolTermMLConfig) -> pd.DataFrame:
    """Vol-Term ML feature 계산 (10 vol features).

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

    close: pd.Series = result["close"]  # type: ignore[assignment]
    high: pd.Series = result["high"]  # type: ignore[assignment]
    low: pd.Series = result["low"]  # type: ignore[assignment]
    open_: pd.Series = result["open"]  # type: ignore[assignment]

    # Returns
    returns = log_returns(close)
    result["returns"] = returns

    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 1-5. Multi-window RV (annualized)
    rv_series: dict[int, pd.Series] = {}
    for w in _RV_WINDOWS:
        rv = realized_volatility(
            returns_series,
            window=w,
            annualization_factor=config.annualization_factor,
        )
        result[f"feat_rv_{w}"] = rv
        rv_series[w] = rv

    # 6-8. Vol Ratio (term structure slope indicators)
    rv_5 = rv_series[5]
    rv_10 = rv_series[10]
    rv_20 = rv_series[20]
    rv_40 = rv_series[40]
    rv_60 = rv_series[60]

    result["feat_vol_ratio_5_20"] = rv_5 / rv_20.clip(lower=1e-10)
    result["feat_vol_ratio_10_40"] = rv_10 / rv_40.clip(lower=1e-10)
    result["feat_vol_ratio_20_60"] = rv_20 / rv_60.clip(lower=1e-10)

    # 9. Parkinson Volatility (high-low based, more precise)
    result["feat_parkinson_vol"] = parkinson_volatility(high, low)

    # 10. Garman-Klass Volatility (OHLC 4가지 사용)
    result["feat_gk_vol"] = garman_klass_volatility(open_, high, low, close)

    # Realized Volatility (for vol_scalar)
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
