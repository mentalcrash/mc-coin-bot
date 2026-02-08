"""Momentum + Mean Reversion Blend Preprocessor (Indicator Calculation).

모멘텀 z-score, 평균회귀 z-score, 실현 변동성 등 기술적 지표를 계산합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
    - #26 VectorBT Standards: Compatible output format
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.mom_mr_blend.config import MomMrBlendConfig

logger = logging.getLogger(__name__)


def calculate_momentum_returns(
    close: pd.Series,
    lookback: int,
) -> pd.Series:
    """모멘텀 수익률 계산 (close / close.shift(lookback) - 1).

    Args:
        close: 종가 시리즈
        lookback: 모멘텀 기간 (캔들 수)

    Returns:
        모멘텀 수익률 시리즈
    """
    mom: pd.Series = close / close.shift(lookback) - 1  # type: ignore[assignment]
    return pd.Series(mom, index=close.index, name="mom_returns")


def calculate_momentum_zscore(
    mom_returns: pd.Series,
    window: int,
) -> pd.Series:
    """모멘텀 수익률의 z-score 계산.

    z = (mom - rolling_mean) / rolling_std

    Args:
        mom_returns: 모멘텀 수익률 시리즈
        window: z-score rolling 윈도우

    Returns:
        모멘텀 z-score 시리즈
    """
    mean = mom_returns.rolling(window=window, min_periods=window).mean()
    std = mom_returns.rolling(window=window, min_periods=window).std()
    z: pd.Series = (mom_returns - mean) / std.replace(0, np.nan)  # type: ignore[assignment]
    return pd.Series(z, index=mom_returns.index, name="mom_zscore")


def calculate_mr_deviation(
    close: pd.Series,
    sma_period: int,
) -> pd.Series:
    """평균회귀 편차 계산 (close - SMA) / SMA.

    Args:
        close: 종가 시리즈
        sma_period: 이동평균 기간

    Returns:
        SMA 대비 편차 비율 시리즈
    """
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()
    deviation: pd.Series = (close - sma) / sma.replace(0, np.nan)  # type: ignore[assignment]
    return pd.Series(deviation, index=close.index, name="mr_deviation")


def calculate_mr_zscore(
    deviation: pd.Series,
    window: int,
) -> pd.Series:
    """평균회귀 편차의 z-score 계산.

    z = (deviation - rolling_mean) / rolling_std

    Args:
        deviation: 편차 시리즈
        window: z-score rolling 윈도우

    Returns:
        평균회귀 z-score 시리즈
    """
    mean = deviation.rolling(window=window, min_periods=window).mean()
    std = deviation.rolling(window=window, min_periods=window).std()
    z: pd.Series = (deviation - mean) / std.replace(0, np.nan)  # type: ignore[assignment]
    return pd.Series(z, index=deviation.index, name="mr_zscore")


def calculate_realized_volatility(
    close: pd.Series,
    window: int,
    annualization_factor: float = 365.0,
) -> pd.Series:
    """실현 변동성 계산 (로그 수익률 기반, 연환산).

    Args:
        close: 종가 시리즈
        window: Rolling 윈도우 크기
        annualization_factor: 연환산 계수

    Returns:
        연환산 변동성 시리즈
    """
    log_returns = np.log(close / close.shift(1))
    rolling_std = log_returns.rolling(window=window, min_periods=window).std()
    return pd.Series(
        rolling_std * np.sqrt(annualization_factor),
        index=close.index,
        name="realized_vol",
    )


def calculate_volatility_scalar(
    realized_vol: pd.Series,
    vol_target: float,
    min_volatility: float = 0.05,
) -> pd.Series:
    """변동성 스케일러 계산 (vol_target / realized_vol).

    shift(1) 적용: 미래 변동성 참조 방지.

    Args:
        realized_vol: 실현 변동성 시리즈
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프

    Returns:
        변동성 스케일러 시리즈 (shift(1) 적용됨)
    """
    clamped_vol = realized_vol.clip(lower=min_volatility)
    scalar = vol_target / clamped_vol
    return pd.Series(scalar.shift(1), index=realized_vol.index, name="vol_scalar")


def preprocess(
    df: pd.DataFrame,
    config: MomMrBlendConfig,
) -> pd.DataFrame:
    """Mom-MR Blend 전처리 (지표 계산).

    OHLCV DataFrame에 블렌드 전략에 필요한 기술적 지표를 추가합니다.

    Calculated Columns:
        - mom_returns: 모멘텀 수익률 (close / close.shift(lookback) - 1)
        - mom_zscore: 모멘텀 z-score
        - mr_deviation: SMA 대비 편차 비율
        - mr_zscore: 평균회귀 z-score
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러 (shift(1) 적용)

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: Mom-MR Blend 설정

    Returns:
        지표가 추가된 새로운 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    required_cols = {"close"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    result = df.copy()

    # Decimal 타입 -> float64 변환
    for col in ["open", "high", "low", "close", "volume"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    close: pd.Series = result["close"]  # type: ignore[assignment]

    # 1. 모멘텀 수익률
    result["mom_returns"] = calculate_momentum_returns(close, config.mom_lookback)

    mom_returns_series: pd.Series = result["mom_returns"]  # type: ignore[assignment]

    # 2. 모멘텀 z-score
    result["mom_zscore"] = calculate_momentum_zscore(
        mom_returns_series,
        window=config.mom_z_window,
    )

    # 3. 평균회귀 편차
    result["mr_deviation"] = calculate_mr_deviation(close, config.mr_lookback)

    mr_deviation_series: pd.Series = result["mr_deviation"]  # type: ignore[assignment]

    # 4. 평균회귀 z-score
    result["mr_zscore"] = calculate_mr_zscore(
        mr_deviation_series,
        window=config.mr_z_window,
    )

    # 5. 실현 변동성
    result["realized_vol"] = calculate_realized_volatility(
        close,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 6. 변동성 스케일러 (shift(1) 내장)
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 지표 통계 로깅
    valid_data = result.dropna(subset=["mom_zscore", "mr_zscore"])
    if len(valid_data) > 0:
        mom_z_mean = valid_data["mom_zscore"].mean()
        mr_z_mean = valid_data["mr_zscore"].mean()
        vs_min = valid_data["vol_scalar"].min()
        vs_max = valid_data["vol_scalar"].max()
        log_msg = (
            "Mom-MR Blend Indicators | Mom Z Mean: %.2f, MR Z Mean: %.2f, Vol Scalar: [%.2f, %.2f]"
        )
        logger.info(log_msg, mom_z_mean, mr_z_mean, vs_min, vs_max)

    return result
