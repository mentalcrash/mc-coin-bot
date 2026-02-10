"""Liquidity-Adjusted Momentum Preprocessor (Indicator Calculation).

Amihud illiquidity, relative volume, momentum 지표를 계산합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
"""

import logging

import numpy as np
import pandas as pd

from src.strategy.liq_momentum.config import LiqMomentumConfig
from src.strategy.vol_regime.preprocessor import (
    calculate_atr,
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)

logger = logging.getLogger(__name__)


def preprocess(
    df: pd.DataFrame,
    config: LiqMomentumConfig,
) -> pd.DataFrame:
    """Liquidity-Adjusted Momentum 전처리 (지표 계산).

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - rel_vol: Relative volume (volume / rolling median)
        - amihud: Amihud illiquidity (|return| / volume)
        - amihud_pctl: Amihud percentile (rolling rank)
        - liq_state: 유동성 상태 (-1=LOW, 0=NORMAL, 1=HIGH)
        - is_weekend: 주말 여부
        - mom_signal: Momentum signal (sign of rolling sum)
        - realized_vol: 실현 변동성
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range
        - drawdown: 롤링 최고점 대비 드로다운

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수, 1H freq)
        config: Liquidity-Adjusted Momentum 설정

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

    result = df.copy()

    # OHLCV 컬럼을 float64로 변환
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    close_series: pd.Series = result["close"]  # type: ignore[assignment]
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]
    volume_series: pd.Series = result["volume"]  # type: ignore[assignment]

    # 1. 수익률 계산
    result["returns"] = calculate_returns(
        close_series,
        use_log=config.use_log_returns,
    )
    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. Relative Volume (volume / rolling median)
    vol_median = volume_series.rolling(
        window=config.rel_vol_window,
        min_periods=config.rel_vol_window,
    ).median()
    vol_median_safe = vol_median.replace(0, np.nan)
    result["rel_vol"] = volume_series / vol_median_safe

    # 3. Amihud Illiquidity (|return| / volume)
    abs_return = returns_series.abs()
    volume_safe = volume_series.clip(lower=1e-10)
    amihud_raw = abs_return / volume_safe
    result["amihud"] = amihud_raw.rolling(
        window=config.amihud_window,
        min_periods=config.amihud_window,
    ).mean()

    # 4. Amihud Percentile (rolling rank)
    amihud_series: pd.Series = result["amihud"]  # type: ignore[assignment]
    result["amihud_pctl"] = amihud_series.rolling(
        window=config.amihud_pctl_window,
        min_periods=config.amihud_pctl_window,
    ).rank(pct=True)

    # 5. Liquidity State classification
    rel_vol_series: pd.Series = result["rel_vol"]  # type: ignore[assignment]
    amihud_pctl_series: pd.Series = result["amihud_pctl"]  # type: ignore[assignment]

    is_low_liq = (rel_vol_series < config.rel_vol_low) | (
        amihud_pctl_series > config.amihud_pctl_high
    )
    is_high_liq = (rel_vol_series > config.rel_vol_high) & (
        amihud_pctl_series < config.amihud_pctl_low
    )

    result["liq_state"] = pd.Series(
        np.where(is_low_liq, -1, np.where(is_high_liq, 1, 0)),
        index=df.index,
    )

    # 6. Weekend flag
    _saturday = 5
    result["is_weekend"] = result.index.dayofweek >= _saturday  # type: ignore[union-attr]

    # 7. Momentum signal
    mom_sum = returns_series.rolling(
        window=config.mom_lookback,
        min_periods=config.mom_lookback,
    ).sum()
    result["mom_signal"] = np.sign(mom_sum)

    # 8. 실현 변동성
    realized_vol = calculate_realized_volatility(
        returns_series,
        window=max(24, config.atr_period),
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = realized_vol

    # 9. 변동성 스케일러
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 10. ATR 계산
    result["atr"] = calculate_atr(
        high_series,
        low_series,
        close_series,
        period=config.atr_period,
    )

    # 11. 드로다운 계산
    result["drawdown"] = calculate_drawdown(close_series)

    # 디버그: 지표 통계
    valid_data = result.dropna(subset=["amihud_pctl"])
    if len(valid_data) > 0:
        low_pct = (valid_data["liq_state"] == -1).mean() * 100
        high_pct = (valid_data["liq_state"] == 1).mean() * 100
        vs_mean = valid_data["vol_scalar"].mean()
        logger.info(
            "Liq-Momentum Indicators | Low Liq: %.1f%%, High Liq: %.1f%%, Avg Vol Scalar: %.4f",
            low_pct,
            high_pct,
            vs_mean,
        )

    return result
