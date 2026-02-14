"""Funding Rate Carry Preprocessor (Indicator Calculation).

이 모듈은 Funding Rate Carry 전략에 필요한 지표를 벡터화된 연산으로 계산합니다.
백테스팅과 라이브 트레이딩 모두에서 동일한 코드를 사용합니다.

Calculated Indicators:
    1. returns: 수익률 (로그 또는 단순)
    2. realized_vol: 실현 변동성 (연환산)
    3. avg_funding_rate: 평균 펀딩비 (rolling mean)
    4. funding_zscore: 펀딩비 Z-score (정규화)
    5. vol_scalar: 변동성 스케일러
    6. atr: Average True Range

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
    - #26 VectorBT Standards: Compatible output format
"""

import logging

import pandas as pd

from src.market.indicators import (
    atr,
    funding_rate_ma,
    funding_zscore,
    log_returns,
    realized_volatility,
    simple_returns,
    volatility_scalar,
)
from src.strategy.funding_carry.config import FundingCarryConfig

logger = logging.getLogger(__name__)


def preprocess(
    df: pd.DataFrame,
    config: FundingCarryConfig,
) -> pd.DataFrame:
    """Funding Rate Carry 전처리 (순수 지표 계산).

    OHLCV + funding_rate DataFrame에 전략에 필요한 기술적 지표를 계산하여 추가합니다.
    모든 계산은 벡터화되어 있으며 for 루프를 사용하지 않습니다.

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - realized_vol: 실현 변동성 (연환산)
        - avg_funding_rate: 평균 펀딩비
        - funding_zscore: 펀딩비 Z-score
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range

    Args:
        df: OHLCV + funding_rate DataFrame (DatetimeIndex 필수)
            필수 컬럼: close, high, low, funding_rate
        config: Funding Carry 설정

    Returns:
        지표가 추가된 새로운 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시

    Example:
        >>> config = FundingCarryConfig(lookback=3, vol_target=0.35)
        >>> processed_df = preprocess(ohlcv_with_funding_df, config)
        >>> processed_df["avg_funding_rate"]
    """
    # 입력 검증
    required_cols = {"close", "high", "low", "funding_rate"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    # 원본 보존 (복사본 생성)
    result = df.copy()

    # OHLCV 컬럼을 float64로 변환 (Decimal 타입 처리)
    numeric_cols = ["open", "high", "low", "close", "volume", "funding_rate"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 컬럼 추출 (명시적 Series 타입)
    close_series: pd.Series = result["close"]  # type: ignore[assignment]
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]
    funding_rate_series: pd.Series = result["funding_rate"]  # type: ignore[assignment]

    # 1. 수익률 계산
    result["returns"] = (
        log_returns(close_series) if config.use_log_returns else simple_returns(close_series)
    )

    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. 실현 변동성 계산 (연환산)
    result["realized_vol"] = realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 3. 평균 펀딩비 계산
    result["avg_funding_rate"] = funding_rate_ma(
        funding_rate_series,
        window=config.lookback,
    )

    # 4. 펀딩비 Z-score 계산
    result["funding_zscore"] = funding_zscore(
        funding_rate_series,
        ma_window=config.lookback,
        zscore_window=config.zscore_window,
    )

    # 5. 변동성 스케일러 계산
    result["vol_scalar"] = volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 6. ATR 계산 (Trailing Stop용 -- 항상 계산)
    result["atr"] = atr(high_series, low_series, close_series)

    # 디버그: 지표 통계 (NaN 제외)
    valid_data = result.dropna()
    if len(valid_data) > 0:
        avg_fr_min = valid_data["avg_funding_rate"].min()
        avg_fr_max = valid_data["avg_funding_rate"].max()
        vs_min = valid_data["vol_scalar"].min()
        vs_max = valid_data["vol_scalar"].max()
        logger.info(
            "Funding Carry Indicators | Avg FR: [%.6f, %.6f], Vol Scalar: [%.2f, %.2f]",
            avg_fr_min,
            avg_fr_max,
            vs_min,
            vs_max,
        )

    return result
