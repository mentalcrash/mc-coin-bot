"""ADX Regime Filter Preprocessor (Indicator Calculation).

이 모듈은 ADX Regime Filter 전략에 필요한 지표를 벡터화된 연산으로 계산합니다.
TSMOM preprocessor의 공통 함수를 재사용합니다.

Calculated Indicators:
    - adx: ADX (Average Directional Index) — 추세 강도
    - returns: 로그 수익률
    - realized_vol: 실현 변동성 (연환산)
    - vol_scalar: 변동성 스케일러 (vol_target / realized_vol)
    - vw_momentum: 거래량 가중 모멘텀
    - z_score: (close - SMA) / std — 평균회귀 시그널
    - drawdown: 롤링 최고점 대비 드로다운
    - atr: Average True Range

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from src.strategy.tsmom.preprocessor import (
    calculate_adx,
    calculate_atr,
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
    calculate_vw_momentum,
)

if TYPE_CHECKING:
    from src.strategy.adx_regime.config import ADXRegimeConfig

logger = logging.getLogger(__name__)


def preprocess(
    df: pd.DataFrame,
    config: ADXRegimeConfig,
) -> pd.DataFrame:
    """ADX Regime Filter 전처리 (지표 계산).

    OHLCV DataFrame에 전략에 필요한 기술적 지표를 계산하여 추가합니다.
    모든 계산은 벡터화되어 있으며 for 루프를 사용하지 않습니다.

    Calculated Columns:
        - adx: ADX (추세 강도, 0~100)
        - returns: 로그 수익률
        - realized_vol: 연환산 변동성
        - vol_scalar: 변동성 스케일러
        - vw_momentum: 거래량 가중 모멘텀
        - z_score: (close - SMA) / std
        - drawdown: 드로다운
        - atr: ATR

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
            필수 컬럼: open, high, low, close, volume
        config: ADX Regime 설정

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
    volume_series: pd.Series = result["volume"]  # type: ignore[assignment]

    # 1. ADX 계산
    result["adx"] = calculate_adx(
        high_series,
        low_series,
        close_series,
        period=config.adx_period,
    )

    # 2. 수익률 계산 (로그)
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

    # 5. 거래량 가중 모멘텀 계산 (momentum leg)
    result["vw_momentum"] = calculate_vw_momentum(
        returns_series,
        volume_series,
        lookback=config.mom_lookback,
    )

    # 6. Z-Score 계산 (mean-reversion leg)
    rolling_mean: pd.Series = close_series.rolling(  # type: ignore[assignment]
        window=config.mr_lookback, min_periods=config.mr_lookback
    ).mean()
    rolling_std: pd.Series = close_series.rolling(  # type: ignore[assignment]
        window=config.mr_lookback, min_periods=config.mr_lookback
    ).std()
    # 0으로 나누기 방지: std가 0이면 NaN
    result["z_score"] = (close_series - rolling_mean) / rolling_std.replace(0, float("nan"))

    # 7. 드로다운 계산 (헤지 숏 모드용)
    result["drawdown"] = calculate_drawdown(close_series)

    # 8. ATR 계산 (Trailing Stop용)
    result["atr"] = calculate_atr(high_series, low_series, close_series)

    # 디버그: 지표 통계 (NaN 제외)
    valid_data = result.dropna()
    if len(valid_data) > 0:
        adx_min = valid_data["adx"].min()
        adx_max = valid_data["adx"].max()
        adx_mean = valid_data["adx"].mean()
        z_min = valid_data["z_score"].min()
        z_max = valid_data["z_score"].max()
        logger.info(
            "ADX Regime Indicators | ADX: [%.1f, %.1f] (avg %.1f), Z-Score: [%.2f, %.2f]",
            adx_min,
            adx_max,
            adx_mean,
            z_min,
            z_max,
        )

    return result
