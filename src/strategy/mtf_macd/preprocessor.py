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

import pandas as pd

from src.market.indicators import (
    log_returns,
    macd,
    realized_volatility,
)

if TYPE_CHECKING:
    from src.strategy.mtf_macd.config import MtfMacdConfig

logger = logging.getLogger(__name__)


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
    macd_line, signal_line, histogram = macd(
        close,
        fast=config.fast_period,
        slow=config.slow_period,
        signal=config.signal_period,
    )
    result["macd_line"] = macd_line
    result["signal_line"] = signal_line
    result["macd_histogram"] = histogram

    # 2. 실현 변동성 계산 (log returns → realized_volatility)
    returns_series = log_returns(close)
    rv = realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = rv

    # 3. 변동성 스케일러 (shift(1): 전봉 변동성 사용, 미래 참조 방지)
    clamped_vol = rv.clip(lower=config.min_volatility)
    prev_vol = clamped_vol.shift(1)
    result["vol_scalar"] = config.vol_target / prev_vol

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
