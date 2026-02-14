"""XSMOM Preprocessor (Indicator Calculation).

이 모듈은 XSMOM 전략에 필요한 지표를 벡터화된 연산으로 계산합니다.
백테스팅과 라이브 트레이딩 모두에서 동일한 코드를 사용합니다.

XSMOM 지표:
    1. rolling_return: lookback 기간 수익률 (log or simple)
    2. realized_vol: 실현 변동성 (연환산)
    3. vol_scalar: 변동성 스케일러 (vol_target / realized_vol)
    4. atr: Average True Range (리스크 관리용)

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
    - #26 VectorBT Standards: Compatible output format
"""

import logging

import numpy as np
import pandas as pd

from src.market.indicators import (
    atr,
    log_returns,
    realized_volatility,
    rolling_return,
    simple_returns,
    volatility_scalar,
)
from src.strategy.xsmom.config import XSMOMConfig

logger = logging.getLogger(__name__)


def calculate_holding_signal(
    signal: pd.Series,
    holding_period: int,
) -> pd.Series:
    """Holding period 기반 시그널 필터링.

    holding_period마다 시그널을 갱신하고, 그 사이에는 이전 시그널을 유지합니다.
    이를 통해 잦은 리밸런싱을 방지합니다.

    Args:
        signal: 원시 시그널 시리즈
        holding_period: 시그널 유지 기간 (캔들 수)

    Returns:
        Holding period가 적용된 시그널 시리즈

    Example:
        >>> held_signal = calculate_holding_signal(raw_signal, holding_period=7)
    """
    if holding_period <= 1:
        return signal.copy()

    # holding_period마다만 시그널을 갱신, 나머지는 ffill
    mask = pd.Series(np.arange(len(signal)) % holding_period == 0, index=signal.index)
    held: pd.Series = signal.where(mask).ffill()  # type: ignore[assignment]
    return pd.Series(held, index=signal.index, name=signal.name)


def preprocess(
    df: pd.DataFrame,
    config: XSMOMConfig,
) -> pd.DataFrame:
    """XSMOM 전처리 (순수 지표 계산).

    OHLCV DataFrame에 XSMOM 전략에 필요한 기술적 지표를 계산하여 추가합니다.
    모든 계산은 벡터화되어 있으며 for 루프를 사용하지 않습니다.

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - realized_vol: 실현 변동성 (연환산)
        - rolling_return: lookback 기간 수익률
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
            필수 컬럼: close, high, low, volume
        config: XSMOM 설정

    Returns:
        지표가 추가된 새로운 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시

    Example:
        >>> config = XSMOMConfig(lookback=21, vol_target=0.35)
        >>> processed_df = preprocess(ohlcv_df, config)
    """
    # 입력 검증
    required_cols = {"close", "high", "low"}
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

    # 1. 수익률 계산 (1-bar returns, 변동성 계산용)
    result["returns"] = (
        log_returns(close_series) if config.use_log_returns else simple_returns(close_series)
    )

    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. 실현 변동성 계산 (연환산)
    result["realized_vol"] = realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 3. Rolling return 계산 (lookback 기간)
    result["rolling_return"] = rolling_return(
        close_series,
        period=config.lookback,
        use_log=config.use_log_returns,
    )

    # 4. 변동성 스케일러 계산
    result["vol_scalar"] = volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 5. ATR 계산 (Trailing Stop용 -- 항상 계산)
    result["atr"] = atr(high_series, low_series, close_series)

    # 디버그: 지표 통계 (NaN 제외)
    valid_data = result.dropna()
    if len(valid_data) > 0:
        rr_min = valid_data["rolling_return"].min()
        rr_max = valid_data["rolling_return"].max()
        vs_min = valid_data["vol_scalar"].min()
        vs_max = valid_data["vol_scalar"].max()
        logger.info(
            "XSMOM Indicators | Rolling Return: [%.4f, %.4f], Vol Scalar: [%.2f, %.2f]",
            rr_min,
            rr_max,
            vs_min,
            vs_max,
        )

    return result
