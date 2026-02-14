"""Entropy Regime Switch Preprocessor (Indicator Calculation).

Shannon Entropy와 ADX를 rolling window로 계산합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

from src.market.indicators import (
    atr,
    drawdown,
    log_returns,
    realized_volatility,
    simple_returns,
    volatility_scalar,
)
from src.strategy.entropy_switch.config import EntropySwitchConfig

logger = logging.getLogger(__name__)


def _rolling_shannon_entropy(
    returns: pd.Series,
    window: int,
    bins: int,
) -> pd.Series:
    """Rolling Shannon Entropy 계산.

    각 윈도우에서 수익률의 히스토그램을 구하고,
    scipy.stats.entropy로 Shannon entropy를 계산합니다.

    Note:
        rolling.apply()는 scipy.stats.entropy를 벡터화할 수 없기 때문에
        이 함수에서만 예외적으로 사용합니다.

    Args:
        returns: 수익률 시리즈
        window: Rolling 윈도우 크기
        bins: 히스토그램 빈 수

    Returns:
        Rolling Shannon entropy 시리즈
    """

    def _compute_entropy(window_data: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> float:
        """단일 윈도우의 Shannon entropy 계산."""
        hist, _ = np.histogram(window_data, bins=bins)
        # +1e-10 prevents log(0)
        return float(scipy_entropy(hist + 1e-10))

    entropy_series: pd.Series = returns.rolling(
        window=window,
        min_periods=window,
    ).apply(_compute_entropy, raw=True)  # type: ignore[assignment]

    return pd.Series(entropy_series, index=returns.index, name="entropy")


def _calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """ADX (Average Directional Index) 계산.

    표준 ADX 공식 (exponential smoothing):
        1. True Range → ATR
        2. +DM, -DM → smoothed +DI, -DI
        3. DX = |+DI - -DI| / (+DI + -DI)
        4. ADX = EWM(DX)

    Args:
        high: 고가 시리즈
        low: 저가 시리즈
        close: 종가 시리즈
        period: 계산 기간

    Returns:
        ADX 시리즈
    """
    # True Range
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)

    # +DM, -DM
    high_diff = high - high.shift(1)
    low_diff = low.shift(1) - low

    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

    # Smoothed TR, +DM, -DM (Wilder's EWM)
    atr_smooth = pd.Series(tr, index=high.index).ewm(alpha=1 / period, min_periods=period).mean()
    plus_dm_smooth = (
        pd.Series(plus_dm, index=high.index, dtype=float)
        .ewm(alpha=1 / period, min_periods=period)
        .mean()
    )
    minus_dm_smooth = (
        pd.Series(minus_dm, index=high.index, dtype=float)
        .ewm(alpha=1 / period, min_periods=period)
        .mean()
    )

    # +DI, -DI
    atr_safe: pd.Series = atr_smooth.replace(0, np.nan)  # type: ignore[assignment]
    plus_di = 100 * plus_dm_smooth / atr_safe
    minus_di = 100 * minus_dm_smooth / atr_safe

    # DX, ADX
    di_sum: pd.Series = (plus_di + minus_di).replace(0, np.nan)  # type: ignore[assignment]
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx: pd.Series = dx.ewm(alpha=1 / period, min_periods=period).mean()  # type: ignore[assignment]

    return pd.Series(adx, index=high.index, name="adx")


def preprocess(
    df: pd.DataFrame,
    config: EntropySwitchConfig,
) -> pd.DataFrame:
    """Entropy Regime Switch 전처리 (지표 계산).

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - entropy: Rolling Shannon entropy
        - mom_direction: 모멘텀 방향 (sign of rolling sum)
        - adx: Average Directional Index
        - realized_vol: 실현 변동성
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range
        - drawdown: 롤링 최고점 대비 드로다운

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: Entropy Switch 설정

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

    # OHLCV 컬럼을 float64로 변환
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    close_series: pd.Series = result["close"]  # type: ignore[assignment]
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]

    # 1. 수익률 계산
    result["returns"] = (
        log_returns(close_series) if config.use_log_returns else simple_returns(close_series)
    )
    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. Rolling Shannon Entropy
    result["entropy"] = _rolling_shannon_entropy(
        returns_series,
        window=config.entropy_window,
        bins=config.entropy_bins,
    )

    # 3. 모멘텀 방향
    mom_sum = returns_series.rolling(
        window=config.mom_lookback,
        min_periods=config.mom_lookback,
    ).sum()
    result["mom_direction"] = np.sign(mom_sum)

    # 4. ADX 계산
    result["adx"] = _calculate_adx(
        high_series,
        low_series,
        close_series,
        period=config.adx_period,
    )

    # 5. 실현 변동성
    realized_vol = realized_volatility(
        returns_series,
        window=config.mom_lookback,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = realized_vol

    # 6. 변동성 스케일러
    result["vol_scalar"] = volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 7. ATR 계산
    result["atr"] = atr(
        high_series,
        low_series,
        close_series,
        period=config.atr_period,
    )

    # 8. 드로다운 계산
    result["drawdown"] = drawdown(close_series)

    # 디버그: 지표 통계
    valid_data = result.dropna()
    if len(valid_data) > 0:
        ent_mean = valid_data["entropy"].mean()
        adx_mean = valid_data["adx"].mean()
        vs_mean = valid_data["vol_scalar"].mean()
        logger.info(
            "Entropy-Switch Indicators | Avg Entropy: %.4f, Avg ADX: %.4f, Avg Vol Scalar: %.4f",
            ent_mean,
            adx_mean,
            vs_mean,
        )

    return result
