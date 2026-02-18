"""On-chain Accumulation Preprocessor — vectorized indicator computation.

Indicators:
    - net_flow: Exchange inflow - outflow (USD)
    - net_flow_zscore: Rolling z-score of net flow → accumulation/distribution
    - stablecoin_roc: Stablecoin total supply ROC → market dry powder
    - oc_mvrv: Market-Value-to-Realized-Value → valuation
    - realized_vol / vol_scalar: Volatility targeting
    - atr: Risk management (TS/SL)

Note: On-chain columns (``oc_mvrv``, ``oc_flow_in_ex_usd``, ``oc_flow_out_ex_usd``,
``oc_stablecoin_total_usd``) are injected by EDA ``StrategyEngine._enrich_onchain()``.
Missing columns default to NaN → score = 0 (neutral).
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.market.indicators import (
    atr,
    log_returns,
    realized_volatility,
    rolling_zscore,
)
from src.strategy.onchain_accum.config import OnchainAccumConfig


def preprocess(df: pd.DataFrame, config: OnchainAccumConfig) -> pd.DataFrame:
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
    has_flow = "oc_flow_in_ex_usd" in result.columns and "oc_flow_out_ex_usd" in result.columns
    has_mvrv = "oc_mvrv" in result.columns
    has_stablecoin = "oc_stablecoin_total_usd" in result.columns

    # 1. Net exchange flow + z-score
    if has_flow:
        flow_in: pd.Series = result["oc_flow_in_ex_usd"]  # type: ignore[assignment]
        flow_out: pd.Series = result["oc_flow_out_ex_usd"]  # type: ignore[assignment]
        net_flow = flow_in - flow_out
        result["net_flow"] = net_flow
        result["net_flow_zscore"] = rolling_zscore(net_flow, window=config.flow_zscore_window)
    else:
        result["net_flow"] = np.nan
        result["net_flow_zscore"] = np.nan

    # 2. MVRV (직접 사용)
    if not has_mvrv:
        result["oc_mvrv"] = np.nan

    # 3. Stablecoin ROC
    if has_stablecoin:
        stab_series: pd.Series = result["oc_stablecoin_total_usd"]  # type: ignore[assignment]
        result["stablecoin_roc"] = stab_series.pct_change(config.stablecoin_roc_window)
    else:
        result["stablecoin_roc"] = np.nan

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
        "Onchain-Accum preprocess | bars={}, has_mvrv={}, has_flow={}, has_stablecoin={}",
        len(result),
        has_mvrv,
        has_flow,
        has_stablecoin,
    )
    return result
