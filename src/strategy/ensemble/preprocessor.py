"""Ensemble Strategy Preprocessor.

vol_scalar를 계산합니다. 서브 전략별 전처리는 각 전략의 run()에서 수행됩니다.

Rules Applied:
    - #12 Data Engineering: Vectorization
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from src.market.indicators import log_returns, realized_volatility

if TYPE_CHECKING:
    from src.strategy.ensemble.config import EnsembleConfig

logger = logging.getLogger(__name__)

_MIN_VOLATILITY = 0.05


def preprocess(
    df: pd.DataFrame,
    config: EnsembleConfig,
) -> pd.DataFrame:
    """앙상블 전처리 — vol_scalar 계산.

    서브 전략의 개별 전처리는 signal.py에서 각 전략의 run() 호출 시 수행됩니다.
    여기서는 앙상블 레벨의 vol_scalar만 계산합니다.

    Calculated Columns:
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러 (shift(1) 적용)

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: 앙상블 설정

    Returns:
        vol_scalar가 추가된 새로운 DataFrame
    """
    required_cols = {"close"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    result = df.copy()

    # float64 변환
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    close_series: pd.Series = result["close"]  # type: ignore[assignment]

    # 실현 변동성
    returns_series = log_returns(close_series)
    rv = realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = rv

    # vol_scalar (shift(1): 전봉 변동성 사용)
    clamped_vol = rv.clip(lower=_MIN_VOLATILITY)
    prev_vol = clamped_vol.shift(1)
    result["vol_scalar"] = config.vol_target / prev_vol

    logger.info(
        "Ensemble Preprocessor | vol_target=%.2f, vol_window=%d",
        config.vol_target,
        config.vol_window,
    )

    return result
