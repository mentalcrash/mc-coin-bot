"""Vol Structure Regime Preprocessor (Indicator Calculation).

이 모듈은 Vol Structure Regime 전략에 필요한 지표를 벡터화된 연산으로 계산합니다.
Short/long volatility ratio와 normalized momentum을 사용하여 regime을 분류합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
    - #26 VectorBT Standards: Compatible output format
"""

import logging

import numpy as np
import pandas as pd

from src.strategy.vol_regime.preprocessor import (
    calculate_atr,
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)
from src.strategy.vol_structure.config import VolStructureConfig

logger = logging.getLogger(__name__)


def preprocess(
    df: pd.DataFrame,
    config: VolStructureConfig,
) -> pd.DataFrame:
    """Vol Structure Regime 전처리 (지표 계산).

    OHLCV DataFrame에 전략에 필요한 기술적 지표를 계산하여 추가합니다.
    모든 계산은 벡터화되어 있으며 for 루프를 사용하지 않습니다.

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - vol_short: 단기 변동성 (연환산)
        - vol_long: 장기 변동성 (연환산)
        - vol_ratio: 단기/장기 변동성 비율
        - norm_momentum: 정규화된 모멘텀 (z-score 유사)
        - realized_vol: 실현 변동성 (vol_short_window 기반)
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range
        - drawdown: 롤링 최고점 대비 드로다운

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
            필수 컬럼: open, high, low, close, volume
        config: Vol Structure Regime 설정

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

    # 원본 보존 (복사본 생성)
    result = df.copy()

    # OHLCV 컬럼을 float64로 변환 (Decimal 타입 처리)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 컬럼 추출
    close_series: pd.Series = result["close"]  # type: ignore[assignment]
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]

    # 1. 수익률 계산
    result["returns"] = calculate_returns(
        close_series,
        use_log=config.use_log_returns,
    )
    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. 단기 변동성 (연환산)
    vol_short = returns_series.rolling(
        window=config.vol_short_window,
        min_periods=config.vol_short_window,
    ).std() * np.sqrt(config.annualization_factor)
    result["vol_short"] = vol_short

    # 3. 장기 변동성 (연환산)
    vol_long = returns_series.rolling(
        window=config.vol_long_window,
        min_periods=config.vol_long_window,
    ).std() * np.sqrt(config.annualization_factor)
    result["vol_long"] = vol_long

    # 4. Vol ratio (단기/장기, 0은 NaN으로 처리하여 division by zero 방지)
    vol_long_safe = vol_long.replace(0, np.nan)
    result["vol_ratio"] = vol_short / vol_long_safe

    # 5. Normalized momentum (returns sum / returns std, z-score 유사)
    returns_sum = returns_series.rolling(
        window=config.mom_window,
        min_periods=config.mom_window,
    ).sum()
    returns_std = returns_series.rolling(
        window=config.mom_window,
        min_periods=config.mom_window,
    ).std()
    returns_std_safe = returns_std.replace(0, np.nan)
    result["norm_momentum"] = returns_sum / returns_std_safe

    # 6. 실현 변동성 (vol_short_window 기반, consistency)
    realized_vol = calculate_realized_volatility(
        returns_series,
        window=config.vol_short_window,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = realized_vol

    # 7. 변동성 스케일러
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 8. ATR 계산 (Trailing Stop용)
    result["atr"] = calculate_atr(
        high_series,
        low_series,
        close_series,
        period=config.atr_period,
    )

    # 9. 드로다운 계산 (헤지 숏 모드용)
    result["drawdown"] = calculate_drawdown(close_series)

    # 디버그: 지표 통계
    valid_data = result.dropna()
    if len(valid_data) > 0:
        vr_mean = valid_data["vol_ratio"].mean()
        nm_mean = valid_data["norm_momentum"].mean()
        vs_mean = valid_data["vol_scalar"].mean()
        logger.info(
            "Vol-Structure Indicators | Avg Vol Ratio: %.4f, Avg Norm Mom: %.4f, Avg Vol Scalar: %.4f",
            vr_mean,
            nm_mean,
            vs_mean,
        )

    return result
