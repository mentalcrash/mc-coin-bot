"""KAMA Trend Following Preprocessor (Indicator Calculation).

KAMA, ATR, 변동성 지표를 벡터화 연산으로 계산합니다.
KAMA 재귀 계산은 numba @njit으로 최적화합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops, except numba)
    - #12 Data Engineering: Log returns for internal calculation
    - #26 VectorBT Standards: Compatible output format
"""

import logging

import pandas as pd

from src.market.indicators import (
    atr,
    drawdown,
    kama,
    log_returns,
    realized_volatility,
    simple_returns,
    volatility_scalar,
)
from src.strategy.kama.config import KAMAConfig

logger = logging.getLogger(__name__)


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

    # Decimal 타입 -> float64 변환
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 컬럼 추출
    close: pd.Series = result["close"]  # type: ignore[assignment]
    high: pd.Series = result["high"]  # type: ignore[assignment]
    low: pd.Series = result["low"]  # type: ignore[assignment]

    # 1. 수익률
    if config.use_log_returns:
        result["returns"] = log_returns(close)
    else:
        result["returns"] = simple_returns(close)

    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. 실현 변동성
    result["realized_vol"] = realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 3. 변동성 스케일러
    result["vol_scalar"] = volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 4. KAMA
    result["kama"] = kama(
        close,
        er_lookback=config.er_lookback,
        fast_period=config.fast_period,
        slow_period=config.slow_period,
    )

    # 5. ATR
    result["atr"] = atr(high, low, close, config.atr_period)

    # 6. 드로다운 (HEDGE_ONLY 모드용)
    result["drawdown"] = drawdown(close)

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
