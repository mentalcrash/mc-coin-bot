"""Candlestick Rejection Momentum Preprocessor (Indicator Calculation).

Bar anatomy + rejection ratio + volume z-score를 벡터화 연산으로 계산합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
"""

import logging

import numpy as np
import pandas as pd

from src.strategy.candle_reject.config import CandleRejectConfig
from src.strategy.vol_regime.preprocessor import (
    calculate_atr,
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)

logger = logging.getLogger(__name__)


def preprocess(
    df: pd.DataFrame,
    config: CandleRejectConfig,
) -> pd.DataFrame:
    """Candlestick Rejection Momentum 전처리 (지표 계산).

    Calculated Columns:
        - upper_wick: 상단 꼬리 길이 (high - max(open, close))
        - lower_wick: 하단 꼬리 길이 (min(open, close) - low)
        - body: 몸통 길이 (abs(close - open))
        - range_: 전체 범위 (high - low)
        - bull_reject: 불 rejection ratio (lower_wick / range)
        - bear_reject: 베어 rejection ratio (upper_wick / range)
        - body_position: 몸통 위치 ((close - low) / range, 0~1)
        - volume_zscore: 거래량 Z-score
        - returns: 수익률 (로그 또는 단순)
        - realized_vol: 실현 변동성
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range
        - drawdown: 롤링 최고점 대비 드로다운

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: Candlestick Rejection 설정

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

    open_series: pd.Series = result["open"]  # type: ignore[assignment]
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]
    close_series: pd.Series = result["close"]  # type: ignore[assignment]
    volume_series: pd.Series = result["volume"]  # type: ignore[assignment]

    # =========================================================================
    # 1. Bar Anatomy (벡터화)
    # =========================================================================
    upper_wick = high_series - np.maximum(open_series, close_series)
    lower_wick = np.minimum(open_series, close_series) - low_series
    body = (close_series - open_series).abs()
    range_ = high_series - low_series

    result["upper_wick"] = upper_wick
    result["lower_wick"] = lower_wick
    result["body"] = body

    # range가 0인 경우 NaN으로 처리 (doji 등)
    range_safe: pd.Series = range_.replace(0, np.nan)  # type: ignore[assignment]
    result["range_"] = range_safe

    # =========================================================================
    # 2. Rejection Ratios
    # =========================================================================
    result["bull_reject"] = lower_wick / range_safe
    result["bear_reject"] = upper_wick / range_safe

    # =========================================================================
    # 3. Body Position: (close - low) / range → 0~1
    #    1에 가까울수록 bullish close, 0에 가까울수록 bearish close
    # =========================================================================
    result["body_position"] = (close_series - low_series) / range_safe

    # =========================================================================
    # 4. Volume Z-score
    # =========================================================================
    vol_mean: pd.Series = volume_series.rolling(  # type: ignore[assignment]
        window=config.volume_zscore_window,
        min_periods=config.volume_zscore_window,
    ).mean()
    vol_std: pd.Series = volume_series.rolling(  # type: ignore[assignment]
        window=config.volume_zscore_window,
        min_periods=config.volume_zscore_window,
    ).std()
    vol_std_safe: pd.Series = vol_std.replace(0, np.nan)  # type: ignore[assignment]
    result["volume_zscore"] = (volume_series - vol_mean) / vol_std_safe

    # =========================================================================
    # 5. 수익률 계산
    # =========================================================================
    result["returns"] = calculate_returns(
        close_series,
        use_log=config.use_log_returns,
    )
    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # =========================================================================
    # 6. 실현 변동성
    # =========================================================================
    realized_vol = calculate_realized_volatility(
        returns_series,
        window=config.volume_zscore_window,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = realized_vol

    # =========================================================================
    # 7. 변동성 스케일러
    # =========================================================================
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # =========================================================================
    # 8. ATR 계산
    # =========================================================================
    result["atr"] = calculate_atr(
        high_series,
        low_series,
        close_series,
        period=config.atr_period,
    )

    # =========================================================================
    # 9. 드로다운 계산
    # =========================================================================
    result["drawdown"] = calculate_drawdown(close_series)

    # 디버그: 지표 통계
    valid_data = result.dropna()
    if len(valid_data) > 0:
        bull_mean = valid_data["bull_reject"].mean()
        bear_mean = valid_data["bear_reject"].mean()
        vs_mean = valid_data["vol_scalar"].mean()
        msg = f"Candle-Reject Indicators | Avg Bull Reject: {bull_mean:.4f}, Avg Bear Reject: {bear_mean:.4f}, Avg Vol Scalar: {vs_mean:.4f}"
        logger.info(msg)

    return result
