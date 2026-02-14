"""Volume Climax Reversal Preprocessor (Indicator Calculation).

Volume Z-score, OBV divergence, close position 등 지표를 계산합니다.

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
from src.strategy.vol_climax.config import VolClimaxConfig

logger = logging.getLogger(__name__)


def _calculate_volume_zscore(
    volume: pd.Series,
    window: int,
) -> pd.Series:
    """Volume Z-score 계산.

    Z-score = (volume - rolling_mean) / rolling_std

    Args:
        volume: 거래량 시리즈
        window: Rolling 윈도우 크기

    Returns:
        Volume Z-score 시리즈
    """
    vol_mean: pd.Series = volume.rolling(  # type: ignore[assignment]
        window=window, min_periods=window
    ).mean()
    vol_std: pd.Series = volume.rolling(  # type: ignore[assignment]
        window=window, min_periods=window
    ).std(ddof=1)

    # std가 0인 경우 NaN 처리
    vol_std_safe: pd.Series = vol_std.replace(0, np.nan)  # type: ignore[assignment]
    return (volume - vol_mean) / vol_std_safe


def _calculate_obv(
    returns: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """On-Balance Volume 계산.

    OBV = cumsum(volume * sign(returns))

    Args:
        returns: 수익률 시리즈
        volume: 거래량 시리즈

    Returns:
        OBV 시리즈
    """
    signed_volume = np.where(returns > 0, volume, np.where(returns < 0, -volume, 0))
    obv = pd.Series(signed_volume, index=volume.index).cumsum()
    return pd.Series(obv, index=volume.index, name="obv")


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


def preprocess(
    df: pd.DataFrame,
    config: VolClimaxConfig,
) -> pd.DataFrame:
    """Volume Climax Reversal 전처리 (지표 계산).

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - volume_zscore: 거래량 Z-score
        - obv: On-Balance Volume
        - obv_direction: OBV 방향 (sign of diff)
        - price_direction: 가격 방향 (sign of diff)
        - divergence: OBV-Price 다이버전스 (bool)
        - close_position: 캔들 내 종가 위치 (0=low, 1=high)
        - realized_vol: 실현 변동성
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range
        - drawdown: 롤링 최고점 대비 드로다운

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: Volume Climax 설정

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
    volume_series: pd.Series = result["volume"]  # type: ignore[assignment]

    # 1. 수익률 계산
    result["returns"] = (
        log_returns(close_series) if config.use_log_returns else simple_returns(close_series)
    )
    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. Volume Z-score
    result["volume_zscore"] = _calculate_volume_zscore(
        volume_series,
        window=config.vol_zscore_window,
    )

    # 3. OBV (On-Balance Volume)
    obv = _calculate_obv(returns_series, volume_series)
    result["obv"] = obv

    # 4. OBV 방향 (lookback 기간 동안의 변화 부호)
    result["obv_direction"] = np.sign(obv.diff(config.obv_lookback))

    # 5. 가격 방향 (lookback 기간 동안의 변화 부호)
    result["price_direction"] = np.sign(close_series.diff(config.obv_lookback))

    # 6. Divergence: OBV 방향과 가격 방향이 다를 때
    obv_dir_series: pd.Series = result["obv_direction"]  # type: ignore[assignment]
    price_dir_series: pd.Series = result["price_direction"]  # type: ignore[assignment]
    result["divergence"] = (obv_dir_series != price_dir_series) & obv_dir_series.notna()

    # 7. Close position in bar range
    result["close_position"] = _calculate_close_position(
        close_series,
        low_series,
        high_series,
    )

    # 8. 실현 변동성
    realized_vol = realized_volatility(
        returns_series,
        window=config.mom_lookback,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = realized_vol

    # 9. 변동성 스케일러
    result["vol_scalar"] = volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 10. ATR 계산
    result["atr"] = atr(
        high_series,
        low_series,
        close_series,
        period=config.atr_period,
    )

    # 11. 드로다운 계산
    result["drawdown"] = drawdown(close_series)

    # 디버그: 지표 통계
    valid_data = result.dropna()
    if len(valid_data) > 0:
        vz_mean = valid_data["volume_zscore"].mean()
        vz_max = valid_data["volume_zscore"].max()
        div_pct = valid_data["divergence"].mean() * 100
        logger.info(
            "Vol-Climax Indicators | Avg Z-score: %.4f, Max Z-score: %.4f, Divergence: %.1f%%",
            vz_mean,
            vz_max,
            div_pct,
        )

    return result
