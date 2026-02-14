"""Capitulation Wick Reversal 전처리 모듈.

OHLCV 데이터에서 ATR spike, volume surge, wick ratio 등 feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from src.strategy.cap_wick_rev.config import CapWickRevConfig

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def _calculate_atr_ratio(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_window: int,
) -> tuple[pd.Series, pd.Series]:
    """ATR spike ratio 계산.

    current_atr / rolling_median_atr로 ATR spike를 감지합니다.

    Args:
        high: 고가 시리즈
        low: 저가 시리즈
        close: 종가 시리즈
        atr_window: Rolling median window

    Returns:
        (current_atr, atr_ratio) 튜플
    """
    # True Range
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Current ATR (단일 바의 TR을 사용하지 않고 14-bar rolling)
    current_atr: pd.Series = tr.rolling(14, min_periods=14).mean()  # type: ignore[assignment]

    # Median ATR (baseline)
    median_atr: pd.Series = current_atr.rolling(  # type: ignore[assignment]
        window=atr_window, min_periods=atr_window
    ).median()
    median_atr_safe: pd.Series = median_atr.replace(0, np.nan)  # type: ignore[assignment]

    atr_ratio = current_atr / median_atr_safe

    return current_atr, atr_ratio


def _calculate_volume_ratio(
    volume: pd.Series,
    window: int,
) -> pd.Series:
    """Volume surge ratio = current_vol / rolling_median_vol.

    Args:
        volume: 거래량 시리즈
        window: Rolling median window

    Returns:
        Volume ratio 시리즈
    """
    median_vol: pd.Series = volume.rolling(  # type: ignore[assignment]
        window=window, min_periods=window
    ).median()
    median_vol_safe: pd.Series = median_vol.replace(0, np.nan)  # type: ignore[assignment]
    return volume / median_vol_safe


def _calculate_wick_ratios(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Upper/lower wick ratio 계산.

    lower_wick_ratio = lower_wick / bar_range
    upper_wick_ratio = upper_wick / bar_range

    Args:
        open_: 시가 시리즈
        high: 고가 시리즈
        low: 저가 시리즈
        close: 종가 시리즈

    Returns:
        (lower_wick_ratio, upper_wick_ratio) 튜플
    """
    bar_range = high - low
    bar_range_safe: pd.Series = bar_range.replace(0, np.nan)  # type: ignore[assignment]

    # Body bottom/top
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    body_high = pd.concat([open_, close], axis=1).max(axis=1)

    lower_wick = body_low - low
    upper_wick = high - body_high

    lower_wick_ratio = lower_wick / bar_range_safe
    upper_wick_ratio = upper_wick / bar_range_safe

    return lower_wick_ratio, upper_wick_ratio


def _calculate_close_position(
    close: pd.Series,
    low: pd.Series,
    high: pd.Series,
) -> pd.Series:
    """Close position within bar range.

    close_position = (close - low) / (high - low)
    0 = near low, 1 = near high

    Args:
        close: 종가 시리즈
        low: 저가 시리즈
        high: 고가 시리즈

    Returns:
        Close position 시리즈 (0~1)
    """
    bar_range = high - low
    bar_range_safe: pd.Series = bar_range.replace(0, np.nan)  # type: ignore[assignment]
    return (close - low) / bar_range_safe


def preprocess(df: pd.DataFrame, config: CapWickRevConfig) -> pd.DataFrame:
    """Capitulation Wick Reversal feature 계산.

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - current_atr: 14-bar rolling ATR
        - atr_ratio: current_atr / rolling_median_atr
        - vol_ratio: current_vol / rolling_median_vol
        - lower_wick_ratio: 하부 wick / bar range
        - upper_wick_ratio: 상부 wick / bar range
        - close_position: 캔들 내 종가 위치 (0=low, 1=high)
        - realized_vol: 실현 변동성
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range (trailing stop용)
        - drawdown: 롤링 최고점 대비 드로다운

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: 전략 설정

    Returns:
        feature가 추가된 새 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    result = df.copy()

    # OHLCV 컬럼을 float64로 변환
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    open_series: pd.Series = result["open"]  # type: ignore[assignment]
    close_series: pd.Series = result["close"]  # type: ignore[assignment]
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]
    volume_series: pd.Series = result["volume"]  # type: ignore[assignment]

    # 1. 수익률 계산
    returns = log_returns(close_series) if config.use_log_returns else simple_returns(close_series)
    result["returns"] = returns

    # 2. ATR ratio (spike detection)
    current_atr, atr_ratio = _calculate_atr_ratio(
        high_series, low_series, close_series, atr_window=config.atr_window
    )
    result["current_atr"] = current_atr
    result["atr_ratio"] = atr_ratio

    # 3. Volume ratio (surge detection)
    result["vol_ratio"] = _calculate_volume_ratio(volume_series, window=config.vol_surge_window)

    # 4. Wick ratios
    lower_wick, upper_wick = _calculate_wick_ratios(
        open_series, high_series, low_series, close_series
    )
    result["lower_wick_ratio"] = lower_wick
    result["upper_wick_ratio"] = upper_wick

    # 5. Close position in bar range
    result["close_position"] = _calculate_close_position(close_series, low_series, high_series)

    # 6. 실현 변동성
    realized_vol = realized_volatility(
        returns,
        window=config.mom_lookback,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = realized_vol

    # 7. 변동성 스케일러
    result["vol_scalar"] = volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 8. ATR (trailing stop용)
    result["atr"] = atr(high_series, low_series, close_series, period=config.atr_period)

    # 9. 드로다운 계산
    result["drawdown"] = drawdown(close_series)

    # 디버그 로깅
    valid_data = result.dropna()
    if len(valid_data) > 0:
        atr_r_mean = valid_data["atr_ratio"].mean()
        vol_r_mean = valid_data["vol_ratio"].mean()
        lw_mean = valid_data["lower_wick_ratio"].mean()
        logger.info(
            "CapWickRev Indicators | Avg ATR Ratio: %.4f, Avg Vol Ratio: %.4f, Avg LW Ratio: %.4f",
            atr_r_mean,
            vol_r_mean,
            lw_mean,
        )

    return result
