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

import pandas as pd

from src.market.indicators import (
    atr,
    donchian_channel,
    log_returns,
    realized_volatility,
)

if TYPE_CHECKING:
    from src.strategy.donchian.config import DonchianConfig

logger = logging.getLogger(__name__)


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

    # 컬럼 추출
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]
    close_series: pd.Series = result["close"]  # type: ignore[assignment]

    # 1. Entry Channel (진입용)
    entry_upper, _entry_mid, entry_lower = donchian_channel(
        high_series, low_series, config.entry_period
    )
    result["entry_upper"] = entry_upper
    result["entry_lower"] = entry_lower

    # 2. Exit Channel (청산용)
    exit_upper, _exit_mid, exit_lower = donchian_channel(
        high_series, low_series, config.exit_period
    )
    result["exit_upper"] = exit_upper
    result["exit_lower"] = exit_lower

    # 3. ATR
    result["atr"] = atr(high_series, low_series, close_series, config.atr_period)

    # 4. 변동성 계산
    returns_series = log_returns(close_series)

    rv = realized_volatility(
        returns_series,
        window=config.atr_period,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = rv

    # 5. 변동성 스케일러
    # Shift(1): 전봉 변동성 사용 (미래 참조 방지)
    clamped_vol = rv.clip(lower=config.min_volatility)
    prev_vol = clamped_vol.shift(1)
    result["vol_scalar"] = config.vol_target / prev_vol

    # 디버그 로깅
    valid_data = result.dropna()
    if len(valid_data) > 0:
        logger.info(
            "Donchian Indicators | Entry Channel: %d, Exit Channel: %d, ATR: %d",
            config.entry_period,
            config.exit_period,
            config.atr_period,
        )

    return result
