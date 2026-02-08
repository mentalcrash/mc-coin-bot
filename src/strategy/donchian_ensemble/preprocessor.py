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

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.donchian_ensemble.config import DonchianEnsembleConfig

logger = logging.getLogger(__name__)


def calculate_donchian_channel(
    high: pd.Series,
    low: pd.Series,
    period: int,
) -> tuple[pd.Series, pd.Series]:
    """Donchian Channel 계산.

    Args:
        high: 고가 Series
        low: 저가 Series
        period: lookback 기간

    Returns:
        (upper, lower) 튜플
        - upper: period 기간 최고가
        - lower: period 기간 최저가
    """
    upper: pd.Series = high.rolling(  # type: ignore[assignment]
        window=period,
        min_periods=period,
    ).max()
    lower: pd.Series = low.rolling(  # type: ignore[assignment]
        window=period,
        min_periods=period,
    ).min()

    return upper, lower


def calculate_realized_volatility(
    close: pd.Series,
    window: int,
    annualization_factor: float,
) -> pd.Series:
    """실현 변동성 계산 (연환산).

    Args:
        close: 종가 Series
        window: Rolling 윈도우
        annualization_factor: 연환산 계수

    Returns:
        연환산 변동성 Series
    """
    log_returns = np.log(close / close.shift(1))

    volatility: pd.Series = log_returns.rolling(  # type: ignore[assignment]
        window=window,
        min_periods=window,
    ).std() * np.sqrt(annualization_factor)

    return volatility


def calculate_volatility_scalar(
    realized_vol: pd.Series,
    vol_target: float,
    min_volatility: float,
) -> pd.Series:
    """변동성 스케일러 계산 (shift(1) 적용).

    strength = vol_target / realized_vol (전봉 기준)

    Args:
        realized_vol: 실현 변동성
        vol_target: 목표 변동성
        min_volatility: 최소 변동성 클램프

    Returns:
        변동성 스케일러 Series (shift(1) 적용됨)
    """
    clamped_vol = realized_vol.clip(lower=min_volatility)

    # Shift(1): 전봉 변동성 사용 (미래 참조 방지)
    prev_vol = clamped_vol.shift(1)

    return vol_target / prev_vol


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
        upper, lower = calculate_donchian_channel(high_series, low_series, lb)
        result[f"dc_upper_{lb}"] = upper
        result[f"dc_lower_{lb}"] = lower

    # 2. 실현 변동성 계산
    realized_vol = calculate_realized_volatility(
        close_series,
        window=config.atr_period,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = realized_vol

    # 3. 변동성 스케일러
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol,
        config.vol_target,
        config.min_volatility,
    )

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
