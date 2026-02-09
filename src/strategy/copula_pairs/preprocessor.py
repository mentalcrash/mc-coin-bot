"""Copula Pairs Trading Preprocessor (Indicator Calculation).

Engle-Granger cointegration 기반 페어 트레이딩 지표를 계산합니다.
Rolling OLS hedge ratio, spread, z-score 등을 벡터화된 연산으로 계산합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops, except rolling OLS)
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
    from src.strategy.copula_pairs.config import CopulaPairsConfig

logger = logging.getLogger(__name__)


def calculate_hedge_ratio(
    close: pd.Series,
    pair_close: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling OLS hedge ratio: close = beta * pair_close + alpha.

    Uses numpy.linalg.lstsq for each rolling window.
    Returns rolling beta series.

    # NOTE: Rolling OLS requires loop -- cannot vectorize arbitrary regression windows

    Args:
        close: 주 자산 종가 시리즈
        pair_close: 페어 자산 종가 시리즈
        window: Rolling 윈도우 크기

    Returns:
        Rolling beta 시리즈

    Raises:
        ValueError: 입력 시리즈 길이 불일치 시
    """
    if len(close) != len(pair_close):
        msg = f"close ({len(close)}) and pair_close ({len(pair_close)}) must have same length"
        raise ValueError(msg)

    n = len(close)
    betas = np.full(n, np.nan)

    # NOTE: Rolling OLS requires loop -- cannot vectorize arbitrary regression windows
    for i in range(window, n):
        y = close.iloc[i - window : i].to_numpy()
        x = pair_close.iloc[i - window : i].to_numpy()
        design_matrix = np.column_stack([x, np.ones(window)])
        try:
            result = np.linalg.lstsq(design_matrix, y, rcond=None)
            betas[i] = result[0][0]
        except np.linalg.LinAlgError:
            betas[i] = np.nan

    return pd.Series(betas, index=close.index, name="hedge_ratio")


def calculate_spread(
    close: pd.Series,
    pair_close: pd.Series,
    hedge_ratio: pd.Series,
) -> pd.Series:
    """Spread = close - beta * pair_close.

    Args:
        close: 주 자산 종가 시리즈
        pair_close: 페어 자산 종가 시리즈
        hedge_ratio: Rolling hedge ratio 시리즈

    Returns:
        Spread 시리즈
    """
    spread: pd.Series = close - hedge_ratio * pair_close  # type: ignore[assignment]
    return pd.Series(spread, index=close.index, name="spread")


def calculate_spread_zscore(
    spread: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling z-score of spread.

    Args:
        spread: Spread 시리즈
        window: Rolling 윈도우 크기

    Returns:
        Spread z-score 시리즈
    """
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    zscore: pd.Series = (spread - mean) / std.replace(0, np.nan)  # type: ignore[assignment]
    return pd.Series(zscore, index=spread.index, name="spread_zscore")


def preprocess(
    df: pd.DataFrame,
    config: CopulaPairsConfig,
) -> pd.DataFrame:
    """Copula Pairs 전처리 (지표 계산).

    OHLCV DataFrame에 페어 트레이딩에 필요한 지표를 계산하여 추가합니다.

    Calculated Columns:
        - returns: 수익률 (로그)
        - realized_vol: 실현 변동성 (연환산)
        - hedge_ratio: Rolling OLS hedge ratio (beta)
        - spread: close - beta * pair_close
        - spread_zscore: Rolling z-score of spread
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
            필수 컬럼: close, high, low, volume, pair_close
        config: CopulaPairs 설정

    Returns:
        지표가 추가된 새로운 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    # 입력 검증
    required_cols = {"close", "high", "low", "volume", "pair_close"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    # 원본 보존 (복사본 생성)
    result = df.copy()

    # OHLCV + pair_close 컬럼을 float64로 변환 (Decimal 타입 처리)
    numeric_cols = ["open", "high", "low", "close", "volume", "pair_close"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 컬럼 추출 (명시적 Series 타입)
    close_series: pd.Series = result["close"]  # type: ignore[assignment]
    pair_close_series: pd.Series = result["pair_close"]  # type: ignore[assignment]
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]

    # 1. 수익률 계산
    result["returns"] = calculate_returns(close_series, use_log=True)

    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. 실현 변동성 계산 (연환산)
    result["realized_vol"] = calculate_realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 3. Hedge ratio 계산 (Rolling OLS)
    result["hedge_ratio"] = calculate_hedge_ratio(
        close_series,
        pair_close_series,
        window=config.formation_window,
    )

    hedge_ratio_series: pd.Series = result["hedge_ratio"]  # type: ignore[assignment]

    # 4. Spread 계산
    result["spread"] = calculate_spread(
        close_series,
        pair_close_series,
        hedge_ratio_series,
    )

    spread_series: pd.Series = result["spread"]  # type: ignore[assignment]

    # 5. Spread z-score 계산
    result["spread_zscore"] = calculate_spread_zscore(
        spread_series,
        window=config.formation_window,
    )

    # 6. 변동성 스케일러 계산
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 7. ATR 계산 (Trailing Stop용)
    result["atr"] = calculate_atr(high_series, low_series, close_series)

    # 디버그: 지표 통계
    valid_data = result.dropna(subset=["spread_zscore"])
    if len(valid_data) > 0:
        zs_min = valid_data["spread_zscore"].min()
        zs_max = valid_data["spread_zscore"].max()
        zs_mean = valid_data["spread_zscore"].mean()
        logger.info(
            "Copula Pairs Indicators | Z-score: [%.4f, %.4f], Mean: %.4f",
            zs_min,
            zs_max,
            zs_mean,
        )

    return result
