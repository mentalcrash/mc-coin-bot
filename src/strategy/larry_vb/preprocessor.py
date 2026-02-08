"""Larry Williams Volatility Breakout Preprocessor.

전일 변동폭 기반 돌파 레벨, 실현 변동성, 변동성 스케일러 등
Larry VB 전략에 필요한 지표를 벡터화 연산으로 계산합니다.

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
    from src.strategy.larry_vb.config import LarryVBConfig

logger = logging.getLogger(__name__)


def calculate_prev_range(
    high: pd.Series,
    low: pd.Series,
) -> pd.Series:
    """전일 변동폭 계산.

    (High - Low)의 전일값을 계산합니다.
    shift(1)로 전일 데이터를 참조하여 lookahead bias를 방지합니다.

    Args:
        high: 고가 시리즈
        low: 저가 시리즈

    Returns:
        전일 변동폭 시리즈 (첫 값은 NaN)
    """
    daily_range = high - low
    return pd.Series(daily_range.shift(1), index=high.index, name="prev_range")


def calculate_breakout_levels(
    open_price: pd.Series,
    prev_range: pd.Series,
    k_factor: float,
) -> tuple[pd.Series, pd.Series]:
    """돌파 레벨 계산.

    당일 시가에 전일 변동폭 * k_factor를 더하거나 빼서
    상단/하단 돌파 레벨을 계산합니다.

    Note:
        breakout_upper/lower는 당일 시가 + 전일 변동폭으로 계산되므로
        장중에도 알 수 있는 값입니다 (lookahead bias 없음).

    Args:
        open_price: 시가 시리즈
        prev_range: 전일 변동폭 시리즈 (shift(1) 적용 완료)
        k_factor: 돌파 배수

    Returns:
        (upper_level, lower_level) 튜플
        - upper_level: Open + k * prev_range
        - lower_level: Open - k * prev_range
    """
    offset = k_factor * prev_range
    upper = pd.Series(open_price + offset, index=open_price.index, name="breakout_upper")
    lower = pd.Series(open_price - offset, index=open_price.index, name="breakout_lower")
    return upper, lower


def calculate_realized_volatility(
    close: pd.Series,
    window: int,
    annualization_factor: float = 365.0,
) -> pd.Series:
    """실현 변동성 계산 (연환산).

    로그 수익률의 Rolling standard deviation을 연환산합니다.

    Args:
        close: 종가 시리즈
        window: Rolling 윈도우 크기
        annualization_factor: 연환산 계수

    Returns:
        연환산 변동성 시리즈
    """
    log_returns = np.log(close / close.shift(1))
    rolling_std = log_returns.rolling(window=window, min_periods=window).std()
    return pd.Series(
        rolling_std * np.sqrt(annualization_factor),
        index=close.index,
        name="realized_vol",
    )


def calculate_volatility_scalar(
    realized_vol: pd.Series,
    vol_target: float,
    min_volatility: float = 0.05,
) -> pd.Series:
    """변동성 스케일러 계산 (vol_target / realized_vol).

    shift(1)을 적용하여 전봉 기준으로 스케일링합니다.
    이는 현재 봉의 변동성은 봉이 완성되어야 알 수 있기 때문입니다.

    Args:
        realized_vol: 실현 변동성 시리즈
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프

    Returns:
        변동성 스케일러 시리즈 (shift(1) 적용)
    """
    clamped_vol = realized_vol.clip(lower=min_volatility)
    raw_scalar = vol_target / clamped_vol
    return pd.Series(raw_scalar.shift(1), index=realized_vol.index, name="vol_scalar")


def preprocess(
    df: pd.DataFrame,
    config: LarryVBConfig,
) -> pd.DataFrame:
    """Larry VB 전략 전처리 (지표 계산).

    OHLCV DataFrame에 Larry VB 전략에 필요한 기술적 지표를 추가합니다.

    Calculated Columns:
        - prev_range: 전일 변동폭 (High - Low).shift(1)
        - breakout_upper: 상단 돌파 레벨 (Open + k * prev_range)
        - breakout_lower: 하단 돌파 레벨 (Open - k * prev_range)
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러 (shift(1) 적용)

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
            필수 컬럼: open, high, low, close
        config: Larry VB 설정

    Returns:
        지표가 추가된 새로운 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    required_cols = {"open", "high", "low", "close"}
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

    # 1. 전일 변동폭
    result["prev_range"] = calculate_prev_range(high_series, low_series)

    prev_range_series: pd.Series = result["prev_range"]  # type: ignore[assignment]

    # 2. 돌파 레벨
    breakout_upper, breakout_lower = calculate_breakout_levels(
        open_series, prev_range_series, config.k_factor
    )
    result["breakout_upper"] = breakout_upper
    result["breakout_lower"] = breakout_lower

    # 3. 실현 변동성 (연환산)
    result["realized_vol"] = calculate_realized_volatility(
        close_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 4. 변동성 스케일러 (shift(1) 포함)
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 지표 통계 로깅
    valid_data = result.dropna()
    if len(valid_data) > 0:
        logger.info(
            "Larry VB Indicators | k_factor: %.2f, Vol Window: %d",
            config.k_factor,
            config.vol_window,
        )

    return result
