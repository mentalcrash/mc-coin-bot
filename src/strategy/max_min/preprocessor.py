"""MAX/MIN Combined Preprocessor (Indicator Calculation).

이 모듈은 MAX/MIN 복합 전략에 필요한 지표를 벡터화된 연산으로 계산합니다.
tsmom.preprocessor의 공통 함수를 재사용합니다.

Calculated Columns:
    - returns: 수익률 (로그)
    - realized_vol: 실현 변동성 (연환산)
    - vol_scalar: 변동성 스케일러 (vol_target / realized_vol)
    - rolling_max: 전봉 기준 lookback 기간 최고가 (Shift(1) 적용)
    - rolling_min: 전봉 기준 lookback 기간 최저가 (Shift(1) 적용)
    - atr: Average True Range

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
    - #26 VectorBT Standards: Compatible output format
    - Shift(1) Rule: rolling_max/min에 shift(1) 적용하여 미래 참조 방지
"""

import logging

import numpy as np
import pandas as pd

from src.market.indicators import (
    atr,
    log_returns,
    realized_volatility,
    volatility_scalar,
)
from src.strategy.max_min.config import MaxMinConfig

logger = logging.getLogger(__name__)


def preprocess(
    df: pd.DataFrame,
    config: MaxMinConfig,
) -> pd.DataFrame:
    """MAX/MIN 전처리 (지표 계산).

    OHLCV DataFrame에 MAX/MIN 전략에 필요한 기술적 지표를 계산하여 추가합니다.
    모든 계산은 벡터화되어 있으며 for 루프를 사용하지 않습니다.

    Shift(1) Rule:
        rolling_max와 rolling_min에 shift(1)을 적용하여
        현재 봉의 고/저점이 시그널 계산에 포함되지 않도록 합니다.

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
            필수 컬럼: open, high, low, close, volume
        config: MAX/MIN 설정

    Returns:
        지표가 추가된 새로운 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    # 입력 검증
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    # 원본 보존 (복사본 생성)
    result = df.copy()

    # OHLCV 컬럼을 float64로 변환 (Decimal 타입 처리)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 컬럼 추출 (명시적 Series 타입)
    close_series: pd.Series = result["close"]  # type: ignore[assignment]
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]

    # 1. 수익률 계산 (로그 수익률)
    result["returns"] = log_returns(close_series)

    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. 실현 변동성 계산 (연환산)
    result["realized_vol"] = realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 3. 변동성 스케일러 계산
    result["vol_scalar"] = volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 4. Rolling MAX/MIN (Shift(1) 적용 — 미래 참조 방지)
    # 전봉까지의 rolling max/min을 사용하여 현재 봉에서 비교
    result["rolling_max"] = (
        high_series.rolling(window=config.lookback, min_periods=config.lookback).max().shift(1)
    )
    result["rolling_min"] = (
        low_series.rolling(window=config.lookback, min_periods=config.lookback).min().shift(1)
    )

    # 5. ATR 계산 (Trailing Stop용 — 항상 계산)
    result["atr"] = atr(high_series, low_series, close_series)

    # 디버그: 지표 통계 로깅
    valid_data = result.dropna()
    if len(valid_data) > 0:
        rmax_last = valid_data["rolling_max"].iloc[-1]
        rmin_last = valid_data["rolling_min"].iloc[-1]
        vs_min = valid_data["vol_scalar"].min()
        vs_max = valid_data["vol_scalar"].max()
        logger.info(
            "MAX/MIN Indicators | Rolling Max: %.2f, Rolling Min: %.2f, Vol Scalar: [%.2f, %.2f]",
            rmax_last,
            rmin_last,
            vs_min,
            vs_max,
        )

        # 방향성 검증
        price_change = float((result["close"].iloc[-1] / result["close"].iloc[0] - 1) * 100)
        close_last = float(result["close"].iloc[-1])
        breakout_pct = (
            (close_last - float(rmax_last)) / float(rmax_last) * 100
            if float(rmax_last) > 0
            else np.nan
        )
        logger.info(
            "Direction Check | Price Change: %+.2f%%, Breakout: %+.2f%%",
            price_change,
            breakout_pct if not np.isnan(breakout_pct) else 0.0,
        )

    return result
