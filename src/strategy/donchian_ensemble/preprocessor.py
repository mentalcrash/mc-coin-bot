"""Donchian Ensemble Preprocessor (Indicator Calculation).

9개 lookback에 대한 Donchian Channel 상/하단과
변동성 스케일러를 계산합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops on DataFrame rows)
    - #12 Data Engineering: Log returns for internal calculation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from src.market.indicators import (
    donchian_channel,
    log_returns,
    realized_volatility,
)

if TYPE_CHECKING:
    from src.strategy.donchian_ensemble.config import DonchianEnsembleConfig

logger = logging.getLogger(__name__)


def preprocess(
    df: pd.DataFrame,
    config: DonchianEnsembleConfig,
) -> pd.DataFrame:
    """Donchian Ensemble 전처리.

    Calculated Columns:
        - dc_upper_{lb}: lookback별 Donchian Channel 상단
        - dc_lower_{lb}: lookback별 Donchian Channel 하단
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러 (shift(1) 적용됨)

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: 전략 설정

    Returns:
        지표가 추가된 새로운 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    # 입력 검증
    required_cols = {"high", "low", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    # 원본 보존
    result = df.copy()

    # OHLCV float64 변환 (Decimal 처리)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 컬럼 추출
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]
    close_series: pd.Series = result["close"]  # type: ignore[assignment]

    # 1. 각 lookback에 대한 Donchian Channel 계산
    for lb in config.lookbacks:
        upper, _middle, lower = donchian_channel(high_series, low_series, lb)
        result[f"dc_upper_{lb}"] = upper
        result[f"dc_lower_{lb}"] = lower

    # 2. 실현 변동성 계산 (log returns → realized_volatility)
    returns_series = log_returns(close_series)
    rv = realized_volatility(
        returns_series,
        window=config.atr_period,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = rv

    # 3. 변동성 스케일러 (shift(1): 전봉 변동성 사용, 미래 참조 방지)
    clamped_vol = rv.clip(lower=config.min_volatility)
    prev_vol = clamped_vol.shift(1)
    result["vol_scalar"] = config.vol_target / prev_vol

    # 디버그 로깅
    valid_data = result.dropna()
    if len(valid_data) > 0:
        vol_min = valid_data["realized_vol"].min()
        vol_max = valid_data["realized_vol"].max()
        logger.info(
            "Donchian Ensemble Indicators | Lookbacks: %d, Volatility: [%.4f, %.4f]",
            len(config.lookbacks),
            vol_min,
            vol_max,
        )

    return result
