"""VW-TSMOM Pure Preprocessor (Indicator Calculation).

이 모듈은 VW-TSMOM Pure 전략에 필요한 지표를 벡터화된 연산으로 계산합니다.
기존 tsmom preprocessor의 헬퍼 함수를 재사용하며,
VW returns 계산을 핵심 시그널로 사용합니다.

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

from src.strategy.tsmom.preprocessor import (
    calculate_atr,
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)

if TYPE_CHECKING:
    from src.strategy.vw_tsmom.config import VWTSMOMConfig

logger = logging.getLogger(__name__)


def calculate_vw_returns(
    returns: pd.Series,
    volume: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """거래량 가중 수익률 계산 (log1p volume for stability).

    각 기간의 수익률에 로그 거래량을 가중하여 평균합니다.
    로그 스케일링으로 거래량 이상치의 과도한 영향력을 압축합니다.

    Formula:
        vw_returns = sum(vol_i * ret_i) / sum(vol_i)
        where vol_i = log1p(volume_i)

    Args:
        returns: 수익률 시리즈
        volume: 거래량 시리즈
        window: Rolling 윈도우 크기
        min_periods: 최소 관측치 수

    Returns:
        거래량 가중 수익률 시리즈

    Example:
        >>> vw_ret = calculate_vw_returns(df["returns"], df["volume"], window=21)
    """
    if min_periods is None:
        min_periods = window

    # log1p 스케일링: ln(1 + volume)로 이상치 영향력 압축
    log_volume = np.log1p(volume)

    # 가중 수익률: sum(return * ln_volume) / sum(ln_volume)
    weighted_returns: pd.Series = (  # type: ignore[assignment]
        (returns * log_volume).rolling(window=window, min_periods=min_periods).sum()
    )
    total_log_volume: pd.Series = log_volume.rolling(  # type: ignore[assignment]
        window=window, min_periods=min_periods
    ).sum()

    # 0으로 나누기 방지
    total_log_volume_safe = total_log_volume.replace(0, np.nan)
    return weighted_returns / total_log_volume_safe


def preprocess(
    df: pd.DataFrame,
    config: VWTSMOMConfig,
) -> pd.DataFrame:
    """VW-TSMOM Pure 전처리 (순수 지표 계산).

    OHLCV DataFrame에 VW-TSMOM Pure 전략에 필요한 기술적 지표를 계산하여 추가합니다.
    모든 계산은 벡터화되어 있으며 for 루프를 사용하지 않습니다.

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - realized_vol: 실현 변동성 (연환산)
        - vw_returns: 거래량 가중 수익률 (VW momentum signal)
        - vol_scalar: 변동성 스케일러
        - drawdown: 롤링 최고점 대비 드로다운
        - atr: Average True Range

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
            필수 컬럼: close, volume
        config: VW-TSMOM 설정

    Returns:
        지표가 추가된 새로운 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시

    Example:
        >>> config = VWTSMOMConfig(lookback=21, vol_target=0.35)
        >>> processed_df = preprocess(ohlcv_df, config)
        >>> processed_df["vw_returns"]  # VW momentum signal
    """
    # 입력 검증
    required_cols = {"close", "volume"}
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
    volume_series: pd.Series = result["volume"]  # type: ignore[assignment]

    # 1. 수익률 계산
    result["returns"] = calculate_returns(
        close_series,
        use_log=config.use_log_returns,
    )

    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. 실현 변동성 계산 (연환산)
    result["realized_vol"] = calculate_realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 3. 거래량 가중 수익률 계산 (VW momentum signal)
    result["vw_returns"] = calculate_vw_returns(
        returns_series,
        volume_series,
        window=config.lookback,
    )

    # 4. 변동성 스케일러 계산
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 5. 드로다운 계산 (헤지 숏 모드용)
    result["drawdown"] = calculate_drawdown(close_series)

    # 6. ATR 계산 (Trailing Stop용 -- 항상 계산)
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]
    result["atr"] = calculate_atr(high_series, low_series, close_series)

    # 디버그: 지표 통계 (NaN 제외)
    valid_data = result.dropna()
    if len(valid_data) > 0:
        vw_min = valid_data["vw_returns"].min()
        vw_max = valid_data["vw_returns"].max()
        vs_min = valid_data["vol_scalar"].min()
        vs_max = valid_data["vol_scalar"].max()
        logger.info(
            "📊 VW-TSMOM Pure Indicators | VW Returns: [%.4f, %.4f], Vol Scalar: [%.2f, %.2f]",
            vw_min,
            vw_max,
            vs_min,
            vs_max,
        )

    return result
