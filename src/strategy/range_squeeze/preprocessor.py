"""Range Compression Breakout Preprocessor (Indicator Calculation).

NR 패턴과 range ratio를 계산하여 vol compression을 감지합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
"""

import logging

import numpy as np
import pandas as pd

from src.market.indicators import (
    atr,
    drawdown,
    log_returns,
    realized_volatility,
    simple_returns,
    volatility_scalar,
)
from src.strategy.range_squeeze.config import RangeSqueezeConfig

logger = logging.getLogger(__name__)


def preprocess(
    df: pd.DataFrame,
    config: RangeSqueezeConfig,
) -> pd.DataFrame:
    """Range Compression Breakout 전처리 (지표 계산).

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - daily_range: 일간 range (high - low)
        - avg_range: 평균 range (rolling mean)
        - range_ratio: 현재 range / 평균 range
        - is_nr: NR 패턴 (N일 중 최소 range)
        - realized_vol: 실현 변동성
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range
        - drawdown: 롤링 최고점 대비 드로다운

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: Range Compression 설정

    Returns:
        지표가 추가된 새로운 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    result = df.copy()

    # OHLCV 컬럼을 float64로 변환
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    close_series: pd.Series = result["close"]  # type: ignore[assignment]
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]

    # 1. 수익률 계산
    result["returns"] = (
        log_returns(close_series) if config.use_log_returns else simple_returns(close_series)
    )
    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. Daily range (dirty data guard: high < low → 0)
    daily_range = (high_series - low_series).clip(lower=0)
    result["daily_range"] = daily_range

    # 3. Average range (rolling mean)
    avg_range = daily_range.rolling(
        window=config.lookback,
        min_periods=config.lookback,
    ).mean()
    result["avg_range"] = avg_range

    # 4. Range ratio (현재 range / 평균 range)
    avg_range_safe = avg_range.replace(0, np.nan)
    result["range_ratio"] = daily_range / avg_range_safe

    # 5. NR 패턴 (N일 중 최소 range 여부)
    rolling_min_range = daily_range.rolling(
        window=config.nr_period,
        min_periods=config.nr_period,
    ).min()
    result["is_nr"] = np.isclose(daily_range, rolling_min_range, rtol=1e-9, equal_nan=False)

    # 6. 실현 변동성
    realized_vol = realized_volatility(
        returns_series,
        window=config.lookback,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = realized_vol

    # 7. 변동성 스케일러
    result["vol_scalar"] = volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 8. ATR 계산
    result["atr"] = atr(
        high_series,
        low_series,
        close_series,
        period=config.atr_period,
    )

    # 9. 드로다운 계산
    result["drawdown"] = drawdown(close_series)

    # 디버그: 지표 통계
    valid_data = result.dropna()
    if len(valid_data) > 0:
        rr_mean = valid_data["range_ratio"].mean()
        nr_pct = valid_data["is_nr"].mean() * 100
        vs_mean = valid_data["vol_scalar"].mean()
        logger.info(
            "Range-Squeeze Indicators | Avg Range Ratio: %.4f, NR%%: %.1f%%, Avg Vol Scalar: %.4f",
            rr_mean,
            nr_pct,
            vs_mean,
        )

    return result
