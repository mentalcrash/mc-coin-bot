"""KAMA Trend Following Preprocessor (Indicator Calculation).

KAMA, ATR, 변동성 지표를 벡터화 연산으로 계산합니다.
KAMA 재귀 계산은 numba @njit으로 최적화합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops, except numba)
    - #12 Data Engineering: Log returns for internal calculation
    - #26 VectorBT Standards: Compatible output format
"""

import logging

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd

from src.strategy.kama.config import KAMAConfig

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


@numba.njit  # type: ignore[misc]
def _compute_kama_numba(
    close_arr: npt.NDArray[np.float64],
    sc_arr: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """KAMA 재귀 계산 (numba 최적화).

    KAMA[i] = KAMA[i-1] + SC[i] * (close[i] - KAMA[i-1])

    Args:
        close_arr: 종가 배열
        sc_arr: Smoothing Constant 배열

    Returns:
        KAMA 값 배열
    """
    n = len(close_arr)
    kama = np.empty(n)
    kama[0] = close_arr[0]
    for i in range(1, n):
        if np.isnan(sc_arr[i]) or np.isnan(close_arr[i]):
            kama[i] = kama[i - 1]
        else:
            kama[i] = kama[i - 1] + sc_arr[i] * (close_arr[i] - kama[i - 1])
    return kama


def calculate_kama(
    close: pd.Series,
    er_lookback: int,
    fast_period: int,
    slow_period: int,
) -> pd.Series:
    """Kaufman Adaptive Moving Average 계산.

    Efficiency Ratio(ER)를 기반으로 시장 상태에 적응하는 이동평균입니다.
    추세가 강할수록(ER -> 1) 빠른 EMA에 가까워지고,
    횡보일수록(ER -> 0) 느린 EMA에 가까워집니다.

    Args:
        close: 종가 시리즈
        er_lookback: Efficiency Ratio 룩백 기간
        fast_period: 빠른 SC 기간
        slow_period: 느린 SC 기간

    Returns:
        KAMA 시리즈
    """
    # Efficiency Ratio = |direction| / volatility
    direction = (close - close.shift(er_lookback)).abs()
    volatility: pd.Series = (
        close.diff()
        .abs()
        .rolling(  # type: ignore[assignment]
            er_lookback, min_periods=er_lookback
        )
        .sum()
    )
    er = direction / volatility.replace(0, np.nan)
    er = er.fillna(0)

    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # Numba 최적화 재귀 계산
    kama_values = _compute_kama_numba(
        close.to_numpy().astype(np.float64),
        sc.to_numpy().astype(np.float64),
    )
    return pd.Series(kama_values, index=close.index, name="kama")


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """ATR (Average True Range) 계산.

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
    config: KAMAConfig,
) -> pd.DataFrame:
    """KAMA 전처리 (지표 계산).

    OHLCV DataFrame에 추세 추종 전략에 필요한 기술적 지표를 추가합니다.

    Calculated Columns:
        - returns: 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - kama: Kaufman Adaptive Moving Average
        - atr: Average True Range
        - drawdown: 최고점 대비 하락률

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: KAMA 설정

    Returns:
        지표가 추가된 새로운 DataFrame
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    result = df.copy()

    # Decimal 타입 → float64 변환
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 컬럼 추출
    close: pd.Series = result["close"]  # type: ignore[assignment]
    high: pd.Series = result["high"]  # type: ignore[assignment]
    low: pd.Series = result["low"]  # type: ignore[assignment]

    # 1. 수익률
    result["returns"] = calculate_returns(close, use_log=config.use_log_returns)

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

    # 4. KAMA
    result["kama"] = calculate_kama(
        close,
        er_lookback=config.er_lookback,
        fast_period=config.fast_period,
        slow_period=config.slow_period,
    )

    # 5. ATR
    result["atr"] = calculate_atr(high, low, close, config.atr_period)

    # 6. 드로다운 (HEDGE_ONLY 모드용)
    result["drawdown"] = calculate_drawdown(close)

    # 지표 통계 로깅
    valid_data = result.dropna()
    if len(valid_data) > 0:
        kama_mean = valid_data["kama"].mean()
        close_mean = valid_data["close"].mean()
        vs_min = valid_data["vol_scalar"].min()
        vs_max = valid_data["vol_scalar"].max()
        logger.info(
            "KAMA Indicators | KAMA Mean: %.1f, Close Mean: %.1f, Vol Scalar: [%.2f, %.2f]",
            kama_mean,
            close_mean,
            vs_min,
            vs_max,
        )

    return result
