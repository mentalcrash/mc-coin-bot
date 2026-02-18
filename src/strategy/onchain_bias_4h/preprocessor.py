"""On-chain Bias 4H 전처리 모듈.

OHLCV + On-chain 데이터에서 phase, er, price_roc, vol_scalar 계산.
"""

import numpy as np
import pandas as pd

from src.market.indicators import (
    atr,
    log_returns,
    realized_volatility,
    volatility_scalar,
)
from src.market.indicators.oscillators import roc
from src.market.indicators.trend import efficiency_ratio
from src.strategy.onchain_bias_4h.config import OnchainBias4hConfig

_REQUIRED_COLUMNS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "oc_mvrv",
        "oc_flow_in_ex_usd",
        "oc_flow_out_ex_usd",
        "oc_stablecoin_total_circulating_usd",
    }
)


def preprocess(df: pd.DataFrame, config: OnchainBias4hConfig) -> pd.DataFrame:
    """On-chain Bias 4H feature 계산.

    Args:
        df: OHLCV + on-chain DataFrame (DatetimeIndex 필수)
        config: 전략 설정

    Returns:
        feature가 추가된 새 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    df = df.copy()

    close: pd.Series = df["close"]  # type: ignore[assignment]
    high: pd.Series = df["high"]  # type: ignore[assignment]
    low: pd.Series = df["low"]  # type: ignore[assignment]

    # --- Returns ---
    returns = log_returns(close)
    df["returns"] = returns

    # --- Realized Volatility ---
    realized_vol = realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    df["realized_vol"] = realized_vol

    # --- Vol Scalar ---
    df["vol_scalar"] = volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # --- On-chain: Net Flow ---
    flow_in: pd.Series = df["oc_flow_in_ex_usd"]  # type: ignore[assignment]
    flow_out: pd.Series = df["oc_flow_out_ex_usd"]  # type: ignore[assignment]
    df["net_flow"] = flow_in - flow_out

    # --- On-chain: Stablecoin ROC ---
    stab: pd.Series = df["oc_stablecoin_total_circulating_usd"]  # type: ignore[assignment]
    df["stab_roc"] = stab.pct_change(config.stab_roc_window)

    # --- On-chain Phase ---
    mvrv: pd.Series = df["oc_mvrv"]  # type: ignore[assignment]
    net_flow: pd.Series = df["net_flow"]  # type: ignore[assignment]
    stab_roc: pd.Series = df["stab_roc"]  # type: ignore[assignment]

    # Phase: ACCUMULATION=1, DISTRIBUTION=-1, NEUTRAL=0
    accumulation = (mvrv < config.mvrv_accumulation) & (net_flow < 0) & (stab_roc > 0)
    distribution = (mvrv > config.mvrv_distribution) & (net_flow > 0) & (stab_roc < 0)
    phase = np.where(accumulation, 1, np.where(distribution, -1, 0))
    df["phase"] = pd.Series(phase, index=df.index, dtype=int)

    # --- Momentum ---
    df["er"] = efficiency_ratio(close, period=config.er_window)
    df["price_roc"] = roc(close, period=config.roc_window)

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    return df
