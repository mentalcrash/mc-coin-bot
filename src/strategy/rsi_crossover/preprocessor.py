"""RSI Crossover Preprocessor (Indicator Calculation).

RSI Crossover 전략에 필요한 지표를 벡터화된 연산으로 계산합니다.
기존 전략 모듈의 함수를 재사용하여 중복을 방지합니다.

Reused Functions:
    - calculate_rsi: BB+RSI 모듈에서 재사용
    - calculate_returns, calculate_realized_volatility,
      calculate_volatility_scalar, calculate_atr: TSMOM 모듈에서 재사용
    - calculate_drawdown: TSMOM 모듈에서 재사용

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
    - #26 VectorBT Standards: Compatible output format
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from src.strategy.bb_rsi.preprocessor import calculate_rsi
from src.strategy.tsmom.preprocessor import (
    calculate_atr,
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)

if TYPE_CHECKING:
    from src.strategy.rsi_crossover.config import RSICrossoverConfig

logger = logging.getLogger(__name__)


def preprocess(
    df: pd.DataFrame,
    config: RSICrossoverConfig,
) -> pd.DataFrame:
    """RSI Crossover 전처리 (지표 계산).

    OHLCV DataFrame에 RSI Crossover 전략에 필요한 기술적 지표를 추가합니다.
    모든 계산은 벡터화되어 있으며 for 루프를 사용하지 않습니다.

    Calculated Columns:
        - rsi: RSI (0-100 범위)
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러 (vol_target / realized_vol)
        - atr: Average True Range
        - drawdown: 최고점 대비 하락률

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
            필수 컬럼: open, high, low, close, volume
        config: RSI Crossover 설정

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

    # 1. RSI 계산
    result["rsi"] = calculate_rsi(close_series, config.rsi_period)

    # 2. 수익률 계산 (로그 수익률)
    result["returns"] = calculate_returns(close_series, use_log=True)

    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 3. 실현 변동성 계산 (연환산)
    result["realized_vol"] = calculate_realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 4. 변동성 스케일러 계산
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 5. ATR 계산 (Trailing Stop용)
    result["atr"] = calculate_atr(high_series, low_series, close_series)

    # 6. 드로다운 계산 (HEDGE_ONLY 모드용)
    result["drawdown"] = calculate_drawdown(close_series)

    # 디버그: 지표 통계 (NaN 제외)
    valid_data = result.dropna()
    if len(valid_data) > 0:
        rsi_mean = valid_data["rsi"].mean()
        rsi_min = valid_data["rsi"].min()
        rsi_max = valid_data["rsi"].max()
        vs_min = valid_data["vol_scalar"].min()
        vs_max = valid_data["vol_scalar"].max()
        logger.info(
            "RSI Crossover Indicators | RSI: [%.1f, %.1f] mean=%.1f, Vol Scalar: [%.2f, %.2f]",
            rsi_min,
            rsi_max,
            rsi_mean,
            vs_min,
            vs_max,
        )

    return result
