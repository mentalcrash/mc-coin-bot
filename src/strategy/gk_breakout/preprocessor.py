"""GK Volatility Breakout Preprocessor.

Garman-Klass variance, vol ratio, Donchian Channel 등
GK Breakout 전략에 필요한 지표를 벡터화 연산으로 계산합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.gk_breakout.config import GKBreakoutConfig

logger = logging.getLogger(__name__)


def calculate_returns(
    close: pd.Series,
    use_log: bool = True,
) -> pd.Series:
    """수익률 계산 (로그 또는 단순).

    Args:
        close: 종가 시리즈
        use_log: True면 로그 수익률, False면 단순 수익률

    Returns:
        수익률 시리즈 (첫 값은 NaN)
    """
    if len(close) == 0:
        msg = "Empty Series provided"
        raise ValueError(msg)

    if use_log:
        price_ratio = close / close.shift(1)
        return pd.Series(np.log(price_ratio), index=close.index, name="returns")
    return close.pct_change()


def calculate_realized_volatility(
    returns: pd.Series,
    window: int,
    annualization_factor: float = 365.0,
) -> pd.Series:
    """실현 변동성 계산 (연환산).

    Args:
        returns: 수익률 시리즈
        window: Rolling 윈도우 크기
        annualization_factor: 연환산 계수

    Returns:
        연환산 변동성 시리즈
    """
    rolling_std = returns.rolling(window=window, min_periods=window).std()
    return rolling_std * np.sqrt(annualization_factor)


def calculate_volatility_scalar(
    realized_vol: pd.Series,
    vol_target: float,
    min_volatility: float = 0.05,
) -> pd.Series:
    """변동성 스케일러 계산 (vol_target / realized_vol).

    Args:
        realized_vol: 실현 변동성 시리즈
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프

    Returns:
        변동성 스케일러 시리즈
    """
    clamped_vol = realized_vol.clip(lower=min_volatility)
    return vol_target / clamped_vol


def calculate_gk_variance(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Garman-Klass single-period variance 계산.

    GK variance는 OHLC 4가지 가격을 모두 활용하여
    close-to-close 변동성보다 효율적인 추정치를 제공합니다.

    Formula:
        gk_var = 0.5 * ln(H/L)^2 - (2*ln2 - 1) * ln(C/O)^2

    Args:
        open_: 시가 시리즈
        high: 고가 시리즈
        low: 저가 시리즈
        close: 종가 시리즈

    Returns:
        GK variance 시리즈 (per-bar)
    """
    ln2 = np.log(2)
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk_var = 0.5 * log_hl**2 - (2 * ln2 - 1) * log_co**2
    return pd.Series(gk_var, index=close.index, name="gk_var")


def calculate_vol_ratio(
    gk_var: pd.Series,
    lookback: int,
) -> pd.Series:
    """단기/장기 GK variance 비율 계산.

    vol_ratio < 1: 변동성 압축 (최근 변동성이 장기보다 낮음)
    vol_ratio > 1: 변동성 확대

    Args:
        gk_var: GK variance 시리즈
        lookback: 단기 윈도우 (장기는 lookback * 2)

    Returns:
        vol ratio 시리즈
    """
    short_vol = gk_var.rolling(lookback, min_periods=lookback).mean()
    long_vol: pd.Series = gk_var.rolling(lookback * 2, min_periods=lookback * 2).mean()  # type: ignore[assignment]
    ratio = short_vol / long_vol.replace(0, np.nan)
    return pd.Series(ratio, index=gk_var.index, name="vol_ratio")


def calculate_donchian_channel(
    high: pd.Series,
    low: pd.Series,
    lookback: int,
) -> tuple[pd.Series, pd.Series]:
    """Donchian Channel 계산.

    NOTE: shift(1)은 signal.py에서 적용됩니다. 여기서는 raw channel만 계산합니다.

    Args:
        high: 고가 시리즈
        low: 저가 시리즈
        lookback: 채널 기간

    Returns:
        (upper, lower) 튜플
        - upper: lookback 기간 고가의 최대값
        - lower: lookback 기간 저가의 최소값
    """
    upper = high.rolling(lookback, min_periods=lookback).max()
    lower = low.rolling(lookback, min_periods=lookback).min()
    return (
        pd.Series(upper, index=high.index, name="dc_upper"),
        pd.Series(lower, index=low.index, name="dc_lower"),
    )


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """ATR (Average True Range) 계산.

    ATR = EWM(True Range, period)
    True Range = max(H-L, |H-Prev_C|, |L-Prev_C|)

    Args:
        high: 고가 시리즈
        low: 저가 시리즈
        close: 종가 시리즈
        period: ATR 기간

    Returns:
        ATR 시리즈
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return pd.Series(atr, index=close.index, name="atr")


def calculate_drawdown(close: pd.Series) -> pd.Series:
    """롤링 최고점 대비 드로다운 계산.

    Args:
        close: 종가 시리즈

    Returns:
        드로다운 시리즈 (항상 0 이하)
    """
    rolling_max = close.expanding().max()
    drawdown: pd.Series = (close - rolling_max) / rolling_max  # type: ignore[assignment]
    return pd.Series(drawdown, index=close.index, name="drawdown")


def preprocess(
    df: pd.DataFrame,
    config: GKBreakoutConfig,
) -> pd.DataFrame:
    """GK Breakout 전략 전처리 (지표 계산).

    OHLCV DataFrame에 GK Breakout 전략에 필요한 기술적 지표를 추가합니다.

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러 (vol_target / realized_vol)
        - gk_var: Garman-Klass variance (per-bar)
        - vol_ratio: 단기/장기 GK variance 비율
        - dc_upper: Donchian Channel 상단
        - dc_lower: Donchian Channel 하단
        - atr: Average True Range
        - drawdown: 최고점 대비 하락률

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: GK Breakout 설정

    Returns:
        지표가 추가된 새로운 DataFrame
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
    open_series: pd.Series = result["open"]  # type: ignore[assignment]
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]
    close_series: pd.Series = result["close"]  # type: ignore[assignment]

    # 1. 수익률
    result["returns"] = calculate_returns(close_series, use_log=config.use_log_returns)

    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. 실현 변동성
    result["realized_vol"] = calculate_realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 3. 변동성 스케일러
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 4. GK Variance
    result["gk_var"] = calculate_gk_variance(open_series, high_series, low_series, close_series)

    gk_var_series: pd.Series = result["gk_var"]  # type: ignore[assignment]

    # 5. Vol Ratio (short/long)
    result["vol_ratio"] = calculate_vol_ratio(gk_var_series, config.gk_lookback)

    # 6. Donchian Channel (raw, shift는 signal.py에서 적용)
    dc_upper, dc_lower = calculate_donchian_channel(
        high_series, low_series, config.breakout_lookback
    )
    result["dc_upper"] = dc_upper
    result["dc_lower"] = dc_lower

    # 7. ATR
    result["atr"] = calculate_atr(high_series, low_series, close_series, config.atr_period)

    # 8. Drawdown (HEDGE_ONLY 모드용)
    result["drawdown"] = calculate_drawdown(close_series)

    # 지표 통계 로깅
    valid_data = result.dropna()
    if len(valid_data) > 0:
        logger.info(
            "GK Breakout Indicators | GK Lookback: %d, Breakout Lookback: %d, Compression: %.2f",
            config.gk_lookback,
            config.breakout_lookback,
            config.compression_threshold,
        )

    return result
