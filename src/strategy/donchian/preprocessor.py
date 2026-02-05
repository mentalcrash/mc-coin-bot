"""Donchian Channel Preprocessor.

Entry/Exit Channel과 ATR 기반 변동성 스케일러를 계산합니다.
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
    from src.strategy.donchian.config import DonchianConfig

logger = logging.getLogger(__name__)


def calculate_entry_channel(
    df: pd.DataFrame,
    period: int,
) -> tuple[pd.Series, pd.Series]:
    """Entry Channel 계산 (진입용).

    Args:
        df: OHLC DataFrame
        period: 채널 기간 (N일)

    Returns:
        (entry_upper, entry_lower) 튜플
        - entry_upper: period일간 고가의 최대값
        - entry_lower: period일간 저가의 최소값
    """
    high: pd.Series = df["high"]  # type: ignore[assignment]
    low: pd.Series = df["low"]  # type: ignore[assignment]

    entry_upper: pd.Series = high.rolling(window=period, min_periods=period).max()  # type: ignore[assignment]

    entry_lower: pd.Series = low.rolling(window=period, min_periods=period).min()  # type: ignore[assignment]

    return entry_upper, entry_lower


def calculate_exit_channel(
    df: pd.DataFrame,
    period: int,
) -> tuple[pd.Series, pd.Series]:
    """Exit Channel 계산 (청산용).

    Args:
        df: OHLC DataFrame
        period: 채널 기간 (M일)

    Returns:
        (exit_upper, exit_lower) 튜플
        - exit_upper: period일간 고가의 최대값 (Short 청산용)
        - exit_lower: period일간 저가의 최소값 (Long 청산용)
    """
    high: pd.Series = df["high"]  # type: ignore[assignment]
    low: pd.Series = df["low"]  # type: ignore[assignment]

    exit_upper: pd.Series = high.rolling(window=period, min_periods=period).max()  # type: ignore[assignment]

    exit_lower: pd.Series = low.rolling(window=period, min_periods=period).min()  # type: ignore[assignment]

    return exit_upper, exit_lower


def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """ATR (Average True Range) 계산.

    ATR = EMA(True Range, period)
    True Range = max(H-L, |H-Prev_C|, |L-Prev_C|)

    Args:
        df: OHLC DataFrame
        period: ATR 계산 기간

    Returns:
        ATR Series
    """
    high: pd.Series = df["high"]  # type: ignore[assignment]
    low: pd.Series = df["low"]  # type: ignore[assignment]
    close: pd.Series = df["close"]  # type: ignore[assignment]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range: pd.Series = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)  # type: ignore[assignment]

    atr: pd.Series = true_range.ewm(span=period, adjust=False).mean()  # type: ignore[assignment]

    return atr


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


def preprocess(df: pd.DataFrame, config: DonchianConfig) -> pd.DataFrame:
    """Donchian 전략 전처리.

    Calculated Columns:
        - entry_upper: Entry Channel 상단 (N일 최고가)
        - entry_lower: Entry Channel 하단 (N일 최저가)
        - exit_upper: Exit Channel 상단 (M일 최고가, Short 청산용)
        - exit_lower: Exit Channel 하단 (M일 최저가, Long 청산용)
        - atr: Average True Range
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러

    Args:
        df: OHLCV DataFrame
        config: 전략 설정

    Returns:
        지표가 추가된 DataFrame
    """
    result = df.copy()

    # OHLCV float64 변환 (Decimal 처리)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 1. Entry Channel (진입용)
    entry_upper, entry_lower = calculate_entry_channel(result, config.entry_period)
    result["entry_upper"] = entry_upper
    result["entry_lower"] = entry_lower

    # 2. Exit Channel (청산용)
    exit_upper, exit_lower = calculate_exit_channel(result, config.exit_period)
    result["exit_upper"] = exit_upper
    result["exit_lower"] = exit_lower

    # 3. ATR
    result["atr"] = calculate_atr(result, config.atr_period)

    # 4. 변동성 계산
    close: pd.Series = result["close"]  # type: ignore[assignment]

    realized_vol = calculate_realized_volatility(
        close,
        window=config.atr_period,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = realized_vol

    # 5. 변동성 스케일러
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol,
        config.vol_target,
        config.min_volatility,
    )

    # 디버그 로깅
    valid_data = result.dropna()
    if len(valid_data) > 0:
        logger.info(
            "📊 Donchian Indicators | Entry Channel: %d, Exit Channel: %d, ATR: %d",
            config.entry_period,
            config.exit_period,
            config.atr_period,
        )

    return result
