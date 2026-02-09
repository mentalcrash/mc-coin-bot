"""Hurst/ER Regime Preprocessor (Indicator Calculation).

Efficiency Ratio, R/S Hurst exponent (numba), momentum, z-score 계산.
모든 계산은 벡터화 또는 numba @njit를 사용합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops except numba @njit)
    - #12 Data Engineering: Log returns for internal calculation
    - #26 VectorBT Standards: Compatible output format
"""

import logging

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd

from src.strategy.hurst_regime.config import HurstRegimeConfig
from src.strategy.tsmom.preprocessor import (
    calculate_atr,
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)

logger = logging.getLogger(__name__)


def calculate_efficiency_ratio(close: pd.Series, lookback: int) -> pd.Series:
    """Kaufman Efficiency Ratio = |direction| / volatility.

    ER이 1에 가까우면 trending, 0에 가까우면 choppy/mean-reverting.

    Args:
        close: 종가 시리즈
        lookback: 계산 기간

    Returns:
        Efficiency Ratio 시리즈 (0~1 범위)
    """
    direction = (close - close.shift(lookback)).abs()
    volatility: pd.Series = close.diff().abs().rolling(lookback, min_periods=lookback).sum()  # type: ignore[assignment]
    er = direction / volatility.replace(0, float("nan"))
    return pd.Series(er.fillna(0), index=close.index, name="er")


@numba.njit  # type: ignore[misc]
def _compute_rolling_hurst_numba(
    returns_arr: npt.NDArray[np.float64],
    window: int,
) -> npt.NDArray[np.float64]:
    """Rolling Hurst exponent via simplified R/S analysis (numba).

    R/S (Rescaled Range) 분석으로 Hurst exponent를 추정합니다.
    H > 0.5: trending (persistent), H < 0.5: mean-reverting (anti-persistent).

    Args:
        returns_arr: 수익률 배열
        window: 분석 윈도우

    Returns:
        Rolling Hurst exponent 배열
    """
    n = len(returns_arr)
    result = np.full(n, np.nan)
    log_window = np.log(window)

    for i in range(window, n):
        segment = returns_arr[i - window : i]

        # Check for NaN
        has_nan = False
        for j in range(window):
            if np.isnan(segment[j]):
                has_nan = True
                break
        if has_nan:
            continue

        # Mean
        mean_seg = 0.0
        for j in range(window):
            mean_seg += segment[j]
        mean_seg /= window

        # Cumulative deviation → rescaled_range
        cum_max = -1e300
        cum_min = 1e300
        cum_sum = 0.0
        for j in range(window):
            cum_sum += segment[j] - mean_seg
            cum_max = max(cum_max, cum_sum)
            cum_min = min(cum_min, cum_sum)
        rescaled_range = cum_max - cum_min

        # Standard deviation → std_dev
        var_sum = 0.0
        for j in range(window):
            diff = segment[j] - mean_seg
            var_sum += diff * diff
        std_dev = np.sqrt(var_sum / window)

        # Hurst = log(R/S) / log(n)
        eps = 1e-15
        if std_dev > eps and rescaled_range > eps:
            result[i] = np.log(rescaled_range / std_dev) / log_window

    return result


def calculate_rolling_hurst(
    returns: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling Hurst exponent via R/S analysis.

    numba @njit 함수를 래핑하여 pd.Series 인터페이스를 제공합니다.

    Args:
        returns: 수익률 시리즈
        window: 분석 윈도우

    Returns:
        Rolling Hurst exponent 시리즈
    """
    returns_arr = returns.to_numpy().astype(np.float64)
    hurst_values = _compute_rolling_hurst_numba(returns_arr, window)
    return pd.Series(hurst_values, index=returns.index, name="hurst")


def preprocess(df: pd.DataFrame, config: HurstRegimeConfig) -> pd.DataFrame:
    """Hurst/ER Regime 전처리 (지표 계산).

    OHLCV DataFrame에 전략에 필요한 기술적 지표를 계산하여 추가합니다.
    모든 계산은 벡터화 또는 numba @njit를 사용합니다.

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - er: Efficiency Ratio (0~1)
        - hurst: Rolling Hurst exponent
        - momentum: 누적 수익률 (trending regime 시그널)
        - z_score: Z-Score (mean-reversion regime 시그널)
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range
        - drawdown: 롤링 최고점 대비 드로다운

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
            필수 컬럼: open, high, low, close, volume
        config: Hurst/ER Regime 설정

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
    result["returns"] = calculate_returns(close_series, use_log=config.use_log_returns)
    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. Efficiency Ratio
    result["er"] = calculate_efficiency_ratio(close_series, config.er_lookback)

    # 3. Rolling Hurst exponent
    result["hurst"] = calculate_rolling_hurst(returns_series, config.hurst_window)

    # 4. Momentum (누적 수익률)
    result["momentum"] = returns_series.rolling(
        config.mom_lookback, min_periods=config.mom_lookback
    ).sum()

    # 5. Z-Score (mean-reversion 시그널)
    rolling_mean: pd.Series = close_series.rolling(
        config.mr_lookback, min_periods=config.mr_lookback
    ).mean()  # type: ignore[assignment]
    rolling_std: pd.Series = close_series.rolling(
        config.mr_lookback, min_periods=config.mr_lookback
    ).std()  # type: ignore[assignment]
    result["z_score"] = (close_series - rolling_mean) / rolling_std.replace(0, float("nan"))

    # 6. 실현 변동성 + vol_scalar
    result["realized_vol"] = calculate_realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 7. ATR + Drawdown
    result["atr"] = calculate_atr(high_series, low_series, close_series, config.atr_period)
    result["drawdown"] = calculate_drawdown(close_series)

    # 디버그: 지표 통계
    valid_data = result.dropna()
    if len(valid_data) > 0:
        er_mean = valid_data["er"].mean()
        hurst_mean = valid_data["hurst"].mean()
        logger.info(
            "Hurst/ER Indicators | ER Mean: %.4f, Hurst Mean: %.4f",
            er_mean,
            hurst_mean,
        )

    return result
