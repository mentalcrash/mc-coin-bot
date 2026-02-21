"""Stablecoin Composition Shift Preprocessor — vectorized indicator computation.

Indicators:
    - usdt_share: USDT / (USDT + USDC) ratio
    - share_roc_short: Short-term ROC of USDT share
    - share_roc_long: Long-term ROC of USDT share
    - realized_vol / vol_scalar: Volatility targeting
    - atr: Risk management (TS/SL)

Note: ``oc_stablecoin_usdt_usd`` and ``oc_stablecoin_usdc_usd`` are injected by
EDA ``StrategyEngine._enrich_onchain()``. Missing columns default to NaN.
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.market.indicators import (
    atr,
    log_returns,
    realized_volatility,
)
from src.strategy.stab_comp.config import StabCompConfig


def preprocess(df: pd.DataFrame, config: StabCompConfig) -> pd.DataFrame:
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

    # On-chain columns (may be absent — graceful degradation)
    has_usdt = "oc_stablecoin_usdt_usd" in result.columns
    has_usdc = "oc_stablecoin_usdc_usd" in result.columns

    # 1. USDT share ratio + ROC
    if has_usdt and has_usdc:
        usdt: pd.Series = result["oc_stablecoin_usdt_usd"].ffill()  # type: ignore[assignment]
        usdc: pd.Series = result["oc_stablecoin_usdc_usd"].ffill()  # type: ignore[assignment]
        total = usdt + usdc
        # Avoid division by zero
        usdt_share = usdt / np.maximum(total, 1.0)
        result["usdt_share"] = usdt_share
        result["share_roc_short"] = usdt_share.pct_change(config.roc_short_window)
        result["share_roc_long"] = usdt_share.pct_change(config.roc_long_window)
    else:
        result["usdt_share"] = np.nan
        result["share_roc_short"] = np.nan
        result["share_roc_long"] = np.nan

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
        "Stab-Comp preprocess | bars={}, has_usdt={}, has_usdc={}",
        len(result),
        has_usdt,
        has_usdc,
    )
    return result
