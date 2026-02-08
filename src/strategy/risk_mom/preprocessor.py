"""Risk-Managed Momentum Preprocessor (Indicator Calculation).

TSMOM 지표를 재사용하되, BSC variance scaling을 추가로 계산합니다.
Barroso-Santa-Clara (2015) 분산 스케일링:
    bsc_scaling = vol_target^2 / max(realized_var, min_variance)

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from src.strategy.tsmom.preprocessor import (
    calculate_atr,
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_vw_momentum,
)

if TYPE_CHECKING:
    from src.strategy.risk_mom.config import RiskMomConfig

logger = logging.getLogger(__name__)


def preprocess(
    df: pd.DataFrame,
    config: RiskMomConfig,
) -> pd.DataFrame:
    """Risk-Managed Momentum 전처리 (지표 계산).

    TSMOM 공통 지표를 재사용하고, BSC variance scaling을 추가합니다.

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - vw_momentum: 거래량 가중 모멘텀
        - realized_vol: 실현 변동성 (연환산)
        - realized_var: 실현 분산 (BSC용, rolling variance)
        - bsc_scaling: BSC 스케일링 = vol_target^2 / clipped(realized_var)
        - drawdown: 롤링 최고점 대비 드로다운
        - atr: Average True Range

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: Risk-Mom 설정

    Returns:
        지표가 추가된 새로운 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    # 입력 검증
    required_cols = {"close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    # 원본 보존
    result = df.copy()

    # OHLCV 컬럼을 float64로 변환 (Decimal 타입 처리)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 컬럼 추출
    close_series: pd.Series = result["close"]  # type: ignore[assignment]
    volume_series: pd.Series = result["volume"]  # type: ignore[assignment]

    # 1. 수익률 계산 (TSMOM 재사용)
    result["returns"] = calculate_returns(
        close_series,
        use_log=config.use_log_returns,
    )

    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. 거래량 가중 모멘텀 계산 (TSMOM 재사용)
    result["vw_momentum"] = calculate_vw_momentum(
        returns_series,
        volume_series,
        lookback=config.lookback,
    )

    # 3. 실현 변동성 계산 (TSMOM 재사용, 로깅용)
    result["realized_vol"] = calculate_realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    # 4. BSC Variance Scaling
    # realized_var = rolling variance of returns
    realized_var: pd.Series = returns_series.rolling(  # type: ignore[assignment]
        window=config.var_window,
    ).var()
    result["realized_var"] = realized_var

    # bsc_scaling = vol_target^2 / max(realized_var, min_variance)
    clamped_var: pd.Series = realized_var.clip(lower=config.min_variance)  # type: ignore[assignment]
    bsc_scaling: pd.Series = config.vol_target**2 / clamped_var  # type: ignore[assignment]
    result["bsc_scaling"] = bsc_scaling

    # 5. 드로다운 계산 (헤지 숏 모드용, TSMOM 재사용)
    result["drawdown"] = calculate_drawdown(close_series)

    # 6. ATR 계산 (Trailing Stop용)
    if {"high", "low"}.issubset(set(df.columns)):
        high_series: pd.Series = result["high"]  # type: ignore[assignment]
        low_series: pd.Series = result["low"]  # type: ignore[assignment]
        result["atr"] = calculate_atr(high_series, low_series, close_series)

    # 디버그 로깅
    valid_data = result.dropna()
    if len(valid_data) > 0:
        mom_min = valid_data["vw_momentum"].min()
        mom_max = valid_data["vw_momentum"].max()
        bsc_min = valid_data["bsc_scaling"].min()
        bsc_max = valid_data["bsc_scaling"].max()
        logger.info(
            "Risk-Mom Indicators | Momentum: [%.4f, %.4f], BSC Scaling: [%.2f, %.2f]",
            mom_min,
            mom_max,
            bsc_min,
            bsc_max,
        )

    return result
