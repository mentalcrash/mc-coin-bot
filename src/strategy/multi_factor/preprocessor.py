"""Multi-Factor Ensemble Preprocessor (Indicator Calculation).

이 모듈은 Multi-Factor Ensemble 전략에 필요한 3개 팩터를 벡터화된 연산으로 계산합니다.
각 팩터는 z-score로 정규화되어 동일 스케일에서 결합됩니다.

Factors:
    1. Momentum Factor: 모멘텀 수익률의 z-score
    2. Volume Shock Factor: 단기/장기 거래량 비율의 z-score
    3. Volatility Factor: 역변동성의 z-score (low vol premium)

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
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)

if TYPE_CHECKING:
    from src.strategy.multi_factor.config import MultiFactorConfig

logger = logging.getLogger(__name__)

# 팩터 수 (모멘텀, 거래량 충격, 역변동성)
NUM_FACTORS = 3


def calculate_momentum_factor(
    close: pd.Series,
    lookback: int,
    zscore_window: int,
) -> pd.Series:
    """모멘텀 팩터 계산 (Rolling Return -> Z-Score).

    지정된 lookback 기간의 수익률을 계산한 후,
    rolling z-score로 정규화하여 시계열 전체에서 비교 가능하게 만듭니다.

    Args:
        close: 종가 시리즈
        lookback: 모멘텀 수익률 계산 기간 (캔들 수)
        zscore_window: z-score 정규화 롤링 윈도우 (캔들 수)

    Returns:
        z-score 정규화된 모멘텀 팩터 시리즈
    """
    rolling_ret = close.pct_change(lookback)
    mean = rolling_ret.rolling(zscore_window).mean()
    std = rolling_ret.rolling(zscore_window).std()
    zscore: pd.Series = (rolling_ret - mean) / std.replace(0, np.nan)  # type: ignore[assignment]
    return pd.Series(zscore, index=close.index, name="momentum_factor")


def calculate_volume_shock_factor(
    volume: pd.Series,
    window: int,
    zscore_window: int,
) -> pd.Series:
    """거래량 충격 팩터 계산 (Abnormal Volume -> Z-Score).

    단기 거래량 평균과 장기 거래량 평균의 차이를 z-score로 정규화합니다.
    비정상적으로 높은 거래량은 가격 변화의 선행 지표로 활용됩니다.

    Args:
        volume: 거래량 시리즈
        window: 단기 거래량 평균 윈도우 (캔들 수)
        zscore_window: 장기 평균 및 표준편차 윈도우 (캔들 수)

    Returns:
        z-score 정규화된 거래량 충격 팩터 시리즈
    """
    short_vol = volume.rolling(window).mean()
    long_vol = volume.rolling(zscore_window).mean()
    long_std = volume.rolling(zscore_window).std()
    zscore: pd.Series = (short_vol - long_vol) / long_std.replace(0, np.nan)  # type: ignore[assignment]
    return pd.Series(zscore, index=volume.index, name="volume_shock_factor")


def calculate_volatility_factor(
    returns: pd.Series,
    window: int,
    zscore_window: int,
) -> pd.Series:
    """역변동성 팩터 계산 (Inverse Volatility Z-Score).

    Rolling 변동성을 z-score로 정규화한 후 부호를 반전합니다.
    낮은 변동성이 높은 점수를 받도록 하여 low volatility premium을 포착합니다.

    Args:
        returns: 수익률 시리즈
        window: 변동성 계산 롤링 윈도우 (캔들 수)
        zscore_window: z-score 정규화 롤링 윈도우 (캔들 수)

    Returns:
        z-score 정규화된 역변동성 팩터 시리즈 (낮은 vol -> 높은 점수)
    """
    rolling_vol = returns.rolling(window).std()
    mean = rolling_vol.rolling(zscore_window).mean()
    std = rolling_vol.rolling(zscore_window).std()
    # Inverse: lower vol gets higher score
    zscore: pd.Series = -(rolling_vol - mean) / std.replace(0, np.nan)  # type: ignore[assignment]
    return pd.Series(zscore, index=returns.index, name="volatility_factor")


def preprocess(
    df: pd.DataFrame,
    config: MultiFactorConfig,
) -> pd.DataFrame:
    """Multi-Factor Ensemble 전처리 (팩터 계산 및 결합).

    OHLCV DataFrame에 3개 팩터와 결합 점수, 변동성 스케일러를 계산하여 추가합니다.
    모든 계산은 벡터화되어 있으며 for 루프를 사용하지 않습니다.

    Calculated Columns:
        - returns: 수익률 (로그)
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러 (vol_target / realized_vol)
        - momentum_factor: 모멘텀 팩터 z-score
        - volume_shock_factor: 거래량 충격 팩터 z-score
        - volatility_factor: 역변동성 팩터 z-score
        - combined_score: 3개 팩터 균등 가중 평균
        - atr: Average True Range (Trailing Stop용)

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
            필수 컬럼: close, high, low, volume
        config: Multi-Factor 설정

    Returns:
        팩터 및 지표가 추가된 새로운 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 또는 빈 DataFrame 시
    """
    required_cols = {"close", "high", "low", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    if df.empty:
        msg = "Input DataFrame is empty"
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
    volume_series: pd.Series = result["volume"]  # type: ignore[assignment]

    # 1. 수익률 계산 (로그 수익률)
    result["returns"] = calculate_returns(close_series, use_log=True)
    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. 실현 변동성 계산 (연환산)
    result["realized_vol"] = calculate_realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 3. 변동성 스케일러 계산
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 4. 모멘텀 팩터 계산
    result["momentum_factor"] = calculate_momentum_factor(
        close_series,
        lookback=config.momentum_lookback,
        zscore_window=config.zscore_window,
    )

    # 5. 거래량 충격 팩터 계산
    result["volume_shock_factor"] = calculate_volume_shock_factor(
        volume_series,
        window=config.volume_shock_window,
        zscore_window=config.zscore_window,
    )

    # 6. 역변동성 팩터 계산
    result["volatility_factor"] = calculate_volatility_factor(
        returns_series,
        window=config.vol_window,
        zscore_window=config.zscore_window,
    )

    # 7. 결합 점수 계산 (균등 가중 평균)
    mom_factor: pd.Series = result["momentum_factor"]  # type: ignore[assignment]
    vol_shock_factor: pd.Series = result["volume_shock_factor"]  # type: ignore[assignment]
    vol_factor: pd.Series = result["volatility_factor"]  # type: ignore[assignment]

    result["combined_score"] = (mom_factor + vol_shock_factor + vol_factor) / NUM_FACTORS

    # 8. ATR 계산 (Trailing Stop용)
    result["atr"] = calculate_atr(high_series, low_series, close_series)

    # 디버그: 팩터 통계
    valid_data = result.dropna()
    if len(valid_data) > 0:
        mom_mean = valid_data["momentum_factor"].mean()
        vs_mean = valid_data["volume_shock_factor"].mean()
        vf_mean = valid_data["volatility_factor"].mean()
        cs_mean = valid_data["combined_score"].mean()
        logger.info(
            "Multi-Factor Indicators | Mom: %.3f, VolShock: %.3f, InvVol: %.3f, Combined: %.3f",
            mom_mean,
            vs_mean,
            vf_mean,
            cs_mean,
        )

    return result
