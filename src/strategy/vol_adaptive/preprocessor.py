"""Vol-Adaptive Trend Preprocessor (Indicator Calculation).

이 모듈은 Vol-Adaptive Trend 전략에 필요한 지표를 벡터화된 연산으로 계산합니다.
EMA crossover, RSI, ADX, 변동성 스케일러 등을 계산합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
    - #26 VectorBT Standards: Compatible output format
"""

import logging

import pandas as pd

from src.market.indicators import (
    adx,
    atr,
    drawdown,
    log_returns,
    realized_volatility,
    simple_returns,
    volatility_scalar,
)
from src.strategy.vol_adaptive.config import VolAdaptiveConfig

logger = logging.getLogger(__name__)


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Relative Strength Index) 계산 (Wilder's method).

    Wilder의 지수이동평균을 사용하여 RSI를 계산합니다.
    0~100 범위의 값을 반환합니다.

    Args:
        close: 종가 시리즈
        period: RSI 계산 기간 (기본 14)

    Returns:
        RSI 시리즈 (0~100 범위)
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's EMA (alpha = 1/period)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi: pd.Series = 100 - (100 / (1 + rs))  # type: ignore[assignment]

    return pd.Series(rsi, index=close.index, name="rsi")


def calculate_ema(
    close: pd.Series,
    span: int,
) -> pd.Series:
    """EMA (Exponential Moving Average) 계산.

    Args:
        close: 종가 시리즈
        span: EMA 기간

    Returns:
        EMA 시리즈
    """
    ema: pd.Series = close.ewm(  # type: ignore[assignment]
        span=span, min_periods=span, adjust=False
    ).mean()
    return ema


def preprocess(
    df: pd.DataFrame,
    config: VolAdaptiveConfig,
) -> pd.DataFrame:
    """Vol-Adaptive Trend 전처리 (지표 계산).

    OHLCV DataFrame에 전략에 필요한 기술적 지표를 계산하여 추가합니다.
    모든 계산은 벡터화되어 있으며 for 루프를 사용하지 않습니다.

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러 (vol_target / realized_vol)
        - ema_fast: 빠른 EMA
        - ema_slow: 느린 EMA
        - rsi: RSI (Wilder's method)
        - adx: ADX (Average Directional Index)
        - atr: Average True Range
        - drawdown: 롤링 최고점 대비 드로다운

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
            필수 컬럼: open, high, low, close, volume
        config: Vol-Adaptive 설정

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

    # 1. 수익률 계산
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

    # 3. 변동성 스케일러 계산
    result["vol_scalar"] = volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 4. EMA (빠른/느린) 계산
    result["ema_fast"] = calculate_ema(close_series, span=config.ema_fast)
    result["ema_slow"] = calculate_ema(close_series, span=config.ema_slow)

    # 5. RSI 계산 (Wilder's method)
    result["rsi"] = calculate_rsi(close_series, period=config.rsi_period)

    # 6. ADX 계산 (추세 강도 필터)
    result["adx"] = adx(
        high_series,
        low_series,
        close_series,
        period=config.adx_period,
    )

    # 7. ATR 계산 (Trailing Stop용)
    result["atr"] = atr(
        high_series,
        low_series,
        close_series,
        period=config.atr_period,
    )

    # 8. 드로다운 계산 (헤지 숏 모드용)
    result["drawdown"] = drawdown(close_series)

    # 디버그: 지표 통계
    valid_data = result.dropna()
    if len(valid_data) > 0:
        rsi_mean = valid_data["rsi"].mean()
        adx_mean = valid_data["adx"].mean()
        vs_min = valid_data["vol_scalar"].min()
        vs_max = valid_data["vol_scalar"].max()
        logger.info(
            "Vol-Adaptive Indicators | RSI Mean: %.2f, ADX Mean: %.2f, Vol Scalar: [%.2f, %.2f]",
            rsi_mean,
            adx_mean,
            vs_min,
            vs_max,
        )

    return result
