"""GK Volatility Breakout Preprocessor.

Garman-Klass variance, vol ratio, Donchian Channel 등
GK Breakout 전략에 필요한 지표를 벡터화 연산으로 계산합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.market.indicators import (
    atr,
    donchian_channel,
    drawdown,
    garman_klass_volatility,
    log_returns,
    realized_volatility,
    simple_returns,
    volatility_scalar,
)

if TYPE_CHECKING:
    from src.strategy.gk_breakout.config import GKBreakoutConfig

logger = logging.getLogger(__name__)


def calculate_vol_ratio(
    gk_var: pd.Series,
    lookback: int,
) -> pd.Series:
    """단기/장기 GK variance 비율 계산.

    vol_ratio < 1: 변동성 압축 (최근 변동성이 장기보다 낮음)
    vol_ratio > 1: 변동성 확대

    Args:
        gk_var: GK variance 시리즈
        lookback: 단기 윈도우 (장기는 lookback * 2)

    Returns:
        vol ratio 시리즈
    """
    short_vol = gk_var.rolling(lookback, min_periods=lookback).mean()
    long_vol: pd.Series = gk_var.rolling(lookback * 2, min_periods=lookback * 2).mean()  # type: ignore[assignment]
    ratio = short_vol / long_vol.replace(0, np.nan)
    return pd.Series(ratio, index=gk_var.index, name="vol_ratio")


def preprocess(
    df: pd.DataFrame,
    config: GKBreakoutConfig,
) -> pd.DataFrame:
    """GK Breakout 전략 전처리 (지표 계산).

    OHLCV DataFrame에 GK Breakout 전략에 필요한 기술적 지표를 추가합니다.

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러 (vol_target / realized_vol)
        - gk_var: Garman-Klass variance (per-bar)
        - vol_ratio: 단기/장기 GK variance 비율
        - dc_upper: Donchian Channel 상단
        - dc_lower: Donchian Channel 하단
        - atr: Average True Range
        - drawdown: 최고점 대비 하락률

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: GK Breakout 설정

    Returns:
        지표가 추가된 새로운 DataFrame
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    result = df.copy()

    # Decimal 타입 -> float64 변환
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 컬럼 추출
    open_series: pd.Series = result["open"]  # type: ignore[assignment]
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]
    close_series: pd.Series = result["close"]  # type: ignore[assignment]

    # 1. 수익률
    if config.use_log_returns:
        result["returns"] = log_returns(close_series)
    else:
        result["returns"] = simple_returns(close_series)

    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. 실현 변동성
    result["realized_vol"] = realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 3. 변동성 스케일러
    result["vol_scalar"] = volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 4. GK Variance
    result["gk_var"] = garman_klass_volatility(open_series, high_series, low_series, close_series)

    gk_var_series: pd.Series = result["gk_var"]  # type: ignore[assignment]

    # 5. Vol Ratio (short/long)
    result["vol_ratio"] = calculate_vol_ratio(gk_var_series, config.gk_lookback)

    # 6. Donchian Channel (raw, shift는 signal.py에서 적용)
    dc_upper, _dc_middle, dc_lower = donchian_channel(
        high_series, low_series, config.breakout_lookback
    )
    result["dc_upper"] = dc_upper
    result["dc_lower"] = dc_lower

    # 7. ATR
    result["atr"] = atr(high_series, low_series, close_series, config.atr_period)

    # 8. Drawdown (HEDGE_ONLY 모드용)
    result["drawdown"] = drawdown(close_series)

    # 지표 통계 로깅
    valid_data = result.dropna()
    if len(valid_data) > 0:
        logger.info(
            "GK Breakout Indicators | GK Lookback: %d, Breakout Lookback: %d, Compression: %.2f",
            config.gk_lookback,
            config.breakout_lookback,
            config.compression_threshold,
        )

    return result
