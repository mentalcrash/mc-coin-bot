"""Overnight Seasonality Preprocessor (Indicator Calculation).

시간대 기반 전략에 필요한 지표를 벡터화된 연산으로 계산합니다.
TSMOM preprocessor의 공통 함수를 재사용합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
    - #26 VectorBT Standards: Compatible output format
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from src.strategy.tsmom.preprocessor import (
    calculate_atr,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)

if TYPE_CHECKING:
    from src.strategy.overnight.config import OvernightConfig

logger = logging.getLogger(__name__)


def preprocess(
    df: pd.DataFrame,
    config: OvernightConfig,
) -> pd.DataFrame:
    """Overnight 전처리 (지표 계산).

    OHLCV DataFrame에 시간대 기반 전략에 필요한 지표를 추가합니다.
    TSMOM preprocessor의 공통 함수를 재사용하여 DRY 원칙을 준수합니다.

    Calculated Columns:
        - hour: DatetimeIndex에서 추출한 시간 (0-23)
        - returns: 수익률 (로그)
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러 (vol_target / realized_vol)
        - atr: Average True Range
        - rolling_vol_ratio: 변동성 비율 (use_vol_filter=True일 때만)

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
            필수 컬럼: open, high, low, close, volume
        config: Overnight 설정

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

    # Decimal 타입 -> float64 변환
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 컬럼 추출
    close: pd.Series = result["close"]  # type: ignore[assignment]
    high: pd.Series = result["high"]  # type: ignore[assignment]
    low: pd.Series = result["low"]  # type: ignore[assignment]

    # 1. Hour 추출 (DatetimeIndex에서)
    result["hour"] = pd.Series(df.index.hour, index=df.index)  # type: ignore[union-attr]

    # 2. 수익률 계산 (로그)
    result["returns"] = calculate_returns(close, use_log=True)

    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 3. 실현 변동성 (연환산)
    result["realized_vol"] = calculate_realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 4. 변동성 스케일러
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 5. ATR (Trailing Stop용)
    result["atr"] = calculate_atr(high, low, close)

    # 6. Volatility Filter (선택적)
    if config.use_vol_filter:
        rolling_mean_vol: pd.Series = realized_vol_series.rolling(  # type: ignore[assignment]
            window=config.vol_filter_lookback,
            min_periods=config.vol_filter_lookback,
        ).mean()
        # 0으로 나누기 방지
        rolling_mean_safe = rolling_mean_vol.clip(lower=config.min_volatility)
        result["rolling_vol_ratio"] = realized_vol_series / rolling_mean_safe

    # 지표 통계 로깅
    valid_data = result.dropna()
    if len(valid_data) > 0:
        vs_min = valid_data["vol_scalar"].min()
        vs_max = valid_data["vol_scalar"].max()
        logger.info(
            "Overnight Indicators | Vol Scalar: [%.2f, %.2f], Entry: %dh, Exit: %dh",
            vs_min,
            vs_max,
            config.entry_hour,
            config.exit_hour,
        )

    return result
