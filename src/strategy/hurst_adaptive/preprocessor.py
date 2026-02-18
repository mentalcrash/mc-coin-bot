"""Hurst-Adaptive Preprocessor — vectorized indicator computation.

Indicators:
    - hurst: Hurst exponent (fractal dimension) → regime detection
    - er: Efficiency ratio (noise/signal) → trend quality
    - trend_momentum: log return rolling sum → momentum direction
    - mr_score: Mean reversion z-score → MR direction
    - realized_vol / vol_scalar: Volatility targeting
    - atr: Risk management (TS/SL)
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.market.indicators import (
    atr,
    efficiency_ratio,
    hurst_exponent,
    log_returns,
    mean_reversion_score,
    realized_volatility,
)
from src.strategy.hurst_adaptive.config import HurstAdaptiveConfig


def preprocess(df: pd.DataFrame, config: HurstAdaptiveConfig) -> pd.DataFrame:
    """지표 사전 계산 (벡터화).

    Args:
        df: OHLCV DataFrame
        config: 전략 설정

    Returns:
        지표 컬럼이 추가된 DataFrame
    """
    required = {"close", "high", "low"}
    missing = required - set(df.columns)
    if missing:
        msg = f"Missing columns: {missing}"
        raise ValueError(msg)

    result = df.copy()
    close_series: pd.Series = result["close"]  # type: ignore[assignment]
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]

    # 1. Hurst exponent — regime detection
    result["hurst"] = hurst_exponent(close_series, window=config.hurst_window)

    # 2. Efficiency ratio — trend quality
    result["er"] = efficiency_ratio(close_series, period=config.er_period)

    # 3. Trend momentum — log return rolling sum
    returns = log_returns(close_series)
    result["trend_momentum"] = returns.rolling(
        window=config.trend_mom_lookback, min_periods=config.trend_mom_lookback
    ).sum()

    # 4. Mean reversion score — z-score 반전
    result["mr_score"] = mean_reversion_score(
        close_series, window=config.mr_window, std_mult=config.mr_std_mult
    )

    # 5. Volatility targeting
    result["realized_vol"] = realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    clamped_vol = np.maximum(result["realized_vol"], config.min_volatility)
    result["vol_scalar"] = config.vol_target / clamped_vol

    # 6. ATR for risk management
    result["atr"] = atr(high_series, low_series, close_series)

    # 7. Drawdown for hedge mode
    cum_max = close_series.cummax()
    result["drawdown"] = (close_series - cum_max) / cum_max

    logger.debug(
        "Hurst-Adaptive preprocess | bars={}, hurst=[{:.3f}, {:.3f}], er=[{:.3f}, {:.3f}]",
        len(result),
        result["hurst"].min(),
        result["hurst"].max(),
        result["er"].min(),
        result["er"].max(),
    )
    return result
