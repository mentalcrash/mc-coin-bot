"""BB+RSI Mean Reversion Preprocessor (Indicator Calculation).

볼린저밴드, RSI, ATR, ADX, 변동성 지표를 벡터화 연산으로 계산합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
    - #26 VectorBT Standards: Compatible output format
"""

import logging

import numpy as np
import pandas as pd

from src.market.indicators import (
    adx as calculate_adx,
    atr as calculate_atr,
    bollinger_bands,
    drawdown as calculate_drawdown,
    log_returns,
    realized_volatility,
    rsi as calculate_rsi,
    simple_returns,
    volatility_scalar,
)
from src.strategy.bb_rsi.config import BBRSIConfig

logger = logging.getLogger(__name__)


def calculate_bb_position(
    close: pd.Series,
    bb_upper: pd.Series,
    bb_lower: pd.Series,
    bb_middle: pd.Series,
) -> pd.Series:
    """볼린저밴드 내 정규화 위치 계산.

    (close - middle) / (upper - lower) → 대략 -0.5 ~ +0.5 범위

    Args:
        close: 종가 시리즈
        bb_upper: 볼린저밴드 상단
        bb_lower: 볼린저밴드 하단
        bb_middle: 볼린저밴드 중간

    Returns:
        BB 내 정규화 위치 시리즈
    """
    bandwidth = bb_upper - bb_lower
    bandwidth_safe = bandwidth.replace(0, np.nan)
    bb_position: pd.Series = (close - bb_middle) / bandwidth_safe  # type: ignore[assignment]
    return pd.Series(bb_position, index=close.index, name="bb_position")


def preprocess(
    df: pd.DataFrame,
    config: BBRSIConfig,
) -> pd.DataFrame:
    """BB+RSI 전처리 (지표 계산).

    OHLCV DataFrame에 평균회귀 전략에 필요한 기술적 지표를 추가합니다.

    Calculated Columns:
        - bb_upper, bb_middle, bb_lower: 볼린저밴드
        - rsi: RSI (0-100)
        - bb_position: BB 내 정규화 위치
        - atr: Average True Range
        - returns: 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - drawdown: 최고점 대비 하락률
        - adx: ADX (use_adx_filter=True일 때)

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: BB+RSI 설정

    Returns:
        지표가 추가된 새로운 DataFrame
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    result = df.copy()

    # Decimal 타입 → float64 변환
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    close: pd.Series = result["close"]  # type: ignore[assignment]
    high: pd.Series = result["high"]  # type: ignore[assignment]
    low: pd.Series = result["low"]  # type: ignore[assignment]

    # 1. 볼린저밴드
    bb_upper, bb_middle, bb_lower = bollinger_bands(close, config.bb_period, config.bb_std)
    result["bb_upper"] = bb_upper
    result["bb_middle"] = bb_middle
    result["bb_lower"] = bb_lower

    # 2. RSI
    result["rsi"] = calculate_rsi(close, config.rsi_period)

    # 3. BB 내 정규화 위치
    result["bb_position"] = calculate_bb_position(close, bb_upper, bb_lower, bb_middle)

    # 4. ATR
    result["atr"] = calculate_atr(high, low, close, config.atr_period)

    # 5. 수익률
    result["returns"] = log_returns(close) if config.use_log_returns else simple_returns(close)

    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 6. 실현 변동성
    result["realized_vol"] = realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 7. 변동성 스케일러
    result["vol_scalar"] = volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 8. 드로다운 (HEDGE_ONLY 모드용)
    result["drawdown"] = calculate_drawdown(close)

    # 9. ADX (레짐 필터)
    if config.use_adx_filter:
        result["adx"] = calculate_adx(high, low, close, period=config.adx_period)

    # 지표 통계 로깅
    valid_data = result.dropna()
    if len(valid_data) > 0:
        rsi_mean = valid_data["rsi"].mean()
        bb_pos_mean = valid_data["bb_position"].mean()
        vs_min = valid_data["vol_scalar"].min()
        vs_max = valid_data["vol_scalar"].max()
        logger.info(
            "📊 BB-RSI Indicators | RSI Mean: %.1f, BB Pos Mean: %.3f, Vol Scalar: [%.2f, %.2f]",
            rsi_mean,
            bb_pos_mean,
            vs_min,
            vs_max,
        )

    return result
