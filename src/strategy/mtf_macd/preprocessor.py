"""MTF MACD Preprocessor (Indicator Calculation).

MACD(12,26,9) 기반 지표를 벡터화된 연산으로 계산합니다.
모든 연산은 벡터화되어 있습니다 (for 루프 금지).

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
    from src.strategy.mtf_macd.config import MtfMacdConfig

logger = logging.getLogger(__name__)


def calculate_macd(
    close: pd.Series,
    fast_period: int,
    slow_period: int,
    signal_period: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD (Moving Average Convergence Divergence) 계산.

    MACD Line = Fast EMA - Slow EMA
    Signal Line = EMA(MACD Line, signal_period)
    Histogram = MACD Line - Signal Line

    Args:
        close: 종가 Series
        fast_period: Fast EMA 기간
        slow_period: Slow EMA 기간
        signal_period: Signal Line EMA 기간

    Returns:
        (macd_line, signal_line, histogram) 튜플
    """
    fast_ema: pd.Series = close.ewm(span=fast_period, adjust=False).mean()  # type: ignore[assignment]
    slow_ema: pd.Series = close.ewm(span=slow_period, adjust=False).mean()  # type: ignore[assignment]

    macd_line = fast_ema - slow_ema

    signal_line: pd.Series = macd_line.ewm(span=signal_period, adjust=False).mean()  # type: ignore[assignment]

    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_realized_volatility(
    close: pd.Series,
    window: int,
    annualization_factor: float,
) -> pd.Series:
    """실현 변동성 계산 (연환산).

    Args:
        close: 종가 Series
        window: Rolling 윈도우
        annualization_factor: 연환산 계수

    Returns:
        연환산 변동성 Series
    """
    log_returns = np.log(close / close.shift(1))

    volatility: pd.Series = log_returns.rolling(window=window, min_periods=window).std() * np.sqrt(
        annualization_factor
    )  # type: ignore[assignment]

    return volatility


def calculate_volatility_scalar(
    realized_vol: pd.Series,
    vol_target: float,
    min_volatility: float,
) -> pd.Series:
    """변동성 스케일러 계산.

    strength = vol_target / realized_vol

    Args:
        realized_vol: 실현 변동성
        vol_target: 목표 변동성
        min_volatility: 최소 변동성 클램프

    Returns:
        변동성 스케일러 Series
    """
    clamped_vol = realized_vol.clip(lower=min_volatility)

    # Shift(1): 전봉 변동성 사용 (미래 참조 방지)
    prev_vol = clamped_vol.shift(1)

    return vol_target / prev_vol


def preprocess(df: pd.DataFrame, config: MtfMacdConfig) -> pd.DataFrame:
    """MTF MACD 전략 전처리.

    OHLCV DataFrame에 MACD 전략에 필요한 기술적 지표를 추가합니다.

    Calculated Columns:
        - macd_line: MACD Line (Fast EMA - Slow EMA)
        - signal_line: Signal Line (EMA of MACD Line)
        - macd_histogram: MACD Histogram (MACD Line - Signal Line)
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
            필수 컬럼: open, high, low, close
        config: 전략 설정

    Returns:
        지표가 추가된 새로운 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    # 입력 검증
    required_cols = {"open", "high", "low", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    # 원본 보존 (복사본 생성)
    result = df.copy()

    # OHLCV float64 변환 (Decimal 처리)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 컬럼 추출
    close: pd.Series = result["close"]  # type: ignore[assignment]

    # 1. MACD 계산
    macd_line, signal_line, histogram = calculate_macd(
        close,
        fast_period=config.fast_period,
        slow_period=config.slow_period,
        signal_period=config.signal_period,
    )
    result["macd_line"] = macd_line
    result["signal_line"] = signal_line
    result["macd_histogram"] = histogram

    # 2. 실현 변동성 계산
    realized_vol = calculate_realized_volatility(
        close,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = realized_vol

    # 3. 변동성 스케일러
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol,
        config.vol_target,
        config.min_volatility,
    )

    # 디버그 로깅
    valid_data = result.dropna()
    if len(valid_data) > 0:
        logger.info(
            "MTF MACD Indicators | MACD(%d,%d,%d), Vol Window: %d",
            config.fast_period,
            config.slow_period,
            config.signal_period,
            config.vol_window,
        )

    return result
