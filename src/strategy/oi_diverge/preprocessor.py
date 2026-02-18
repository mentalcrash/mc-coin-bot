"""OI-Price Divergence Preprocessor — vectorized indicator computation.

Indicators:
    - oi_price_div: OI-price rolling correlation → divergence detection
    - fr_zscore: Funding rate z-score → crowding detection
    - oi_roc: OI momentum → OI trend
    - realized_vol / vol_scalar: Volatility targeting
    - atr: Risk management (TS/SL)

Note: ``funding_rate`` and ``open_interest`` columns are injected by
EDA ``StrategyEngine._enrich_derivatives()``.
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.market.indicators import (
    atr,
    funding_zscore,
    log_returns,
    oi_momentum,
    oi_price_divergence,
    realized_volatility,
)
from src.strategy.oi_diverge.config import OiDivergeConfig


def preprocess(df: pd.DataFrame, config: OiDivergeConfig) -> pd.DataFrame:
    """지표 사전 계산 (벡터화).

    Args:
        df: OHLCV + derivatives DataFrame
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

    # Derivatives columns (may be missing in pure OHLCV mode)
    has_derivatives = "funding_rate" in result.columns and "open_interest" in result.columns

    if has_derivatives:
        fr_series: pd.Series = result["funding_rate"]  # type: ignore[assignment]
        oi_series: pd.Series = result["open_interest"]  # type: ignore[assignment]

        # 1. OI-price divergence (rolling correlation)
        result["oi_price_div"] = oi_price_divergence(
            close_series, oi_series, window=config.divergence_window
        )

        # 2. Funding rate z-score
        result["fr_zscore"] = funding_zscore(
            fr_series,
            ma_window=config.fr_ma_window,
            zscore_window=config.fr_zscore_window,
        )

        # 3. OI momentum (rate of change)
        result["oi_roc"] = oi_momentum(oi_series, period=config.oi_momentum_period)
    else:
        # Neutral values — no signal when derivatives missing
        result["oi_price_div"] = 0.0
        result["fr_zscore"] = 0.0
        result["oi_roc"] = 0.0

    # 4. Volatility targeting
    returns = log_returns(close_series)
    result["realized_vol"] = realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    clamped_vol = np.maximum(result["realized_vol"], config.min_volatility)
    result["vol_scalar"] = config.vol_target / clamped_vol

    # 5. ATR for risk management
    result["atr"] = atr(high_series, low_series, close_series)

    logger.debug(
        "OI-Diverge preprocess | bars={}, has_derivatives={}, div=[{:.3f}, {:.3f}], fr_z=[{:.3f}, {:.3f}]",
        len(result),
        has_derivatives,
        result["oi_price_div"].min(),
        result["oi_price_div"].max(),
        result["fr_zscore"].min(),
        result["fr_zscore"].max(),
    )
    return result
