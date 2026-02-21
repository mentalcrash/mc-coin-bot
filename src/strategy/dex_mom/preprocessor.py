"""DEX Activity Momentum Preprocessor — vectorized indicator computation.

Indicators:
    - dex_roc_short: Short-term ROC of DEX volume
    - dex_roc_long: Long-term ROC of DEX volume
    - realized_vol / vol_scalar: Volatility targeting
    - atr: Risk management (TS/SL)

Note: ``oc_dex_volume_usd`` is injected by EDA ``StrategyEngine._enrich_onchain()``.
Missing column defaults to NaN → score = 0 (neutral).
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.market.indicators import (
    atr,
    log_returns,
    realized_volatility,
)
from src.strategy.dex_mom.config import DexMomConfig


def preprocess(df: pd.DataFrame, config: DexMomConfig) -> pd.DataFrame:
    """지표 사전 계산 (벡터화).

    Args:
        df: OHLCV + on-chain DataFrame
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

    # On-chain column (may be absent — graceful degradation)
    has_dex = "oc_dex_volume_usd" in result.columns

    # 1. DEX volume ROC
    if has_dex:
        dex_vol: pd.Series = result["oc_dex_volume_usd"].ffill()  # type: ignore[assignment]
        result["dex_roc_short"] = dex_vol.pct_change(config.roc_short_window)
        result["dex_roc_long"] = dex_vol.pct_change(config.roc_long_window)
    else:
        result["dex_roc_short"] = np.nan
        result["dex_roc_long"] = np.nan

    # 2. Volatility targeting
    returns = log_returns(close_series)
    result["realized_vol"] = realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    clamped_vol = np.maximum(result["realized_vol"], config.min_volatility)
    result["vol_scalar"] = config.vol_target / clamped_vol

    # 3. ATR for risk management
    result["atr"] = atr(high_series, low_series, close_series)

    logger.debug(
        "Dex-Mom preprocess | bars={}, has_dex={}",
        len(result),
        has_dex,
    )
    return result
