"""Liquidity-Confirmed Trend 전처리 모듈.

OHLCV + on-chain (stablecoin, TVL, F&G) 데이터에서 전략 feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).

On-chain columns (oc_stablecoin_total_usd, oc_tvl_usd, oc_fear_greed)은
EDA StrategyEngine._enrich_onchain()이 주입한다.
Missing columns → NaN → score = 0 (neutral).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.liq_conf_trend.config import LiqConfTrendConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    rolling_return,
    sma,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: LiqConfTrendConfig) -> pd.DataFrame:
    """Liquidity-Confirmed Trend feature 계산.

    Calculated columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - price_mom: 가격 모멘텀 (rolling return)
        - stablecoin_roc: Stablecoin supply ROC (on-chain, optional)
        - tvl_roc: TVL ROC (on-chain, optional)
        - liq_score: Liquidity composite score (0~2)
        - fg_ma: F&G smoothed (SMA, optional)
        - drawdown: rolling drawdown (HEDGE_ONLY용)

    Args:
        df: OHLCV + optional on-chain DataFrame (DatetimeIndex 필수)
        config: 전략 설정

    Returns:
        feature가 추가된 새 DataFrame

    Raises:
        ValueError: 필수 OHLCV 컬럼 누락 시
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    df = df.copy()

    close: pd.Series = df["close"]  # type: ignore[assignment]

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

    # --- Price Momentum ---
    df["price_mom"] = rolling_return(close, period=config.mom_lookback)

    # --- Stablecoin Supply ROC (optional, graceful degradation) ---
    has_stablecoin = "oc_stablecoin_total_usd" in df.columns
    if has_stablecoin:
        stab: pd.Series = df["oc_stablecoin_total_usd"]  # type: ignore[assignment]
        stab = stab.ffill()
        df["stablecoin_roc"] = stab.pct_change(config.stablecoin_roc_window)
    else:
        df["stablecoin_roc"] = np.nan

    # --- TVL ROC (optional, graceful degradation) ---
    has_tvl = "oc_tvl_usd" in df.columns
    if has_tvl:
        tvl: pd.Series = df["oc_tvl_usd"]  # type: ignore[assignment]
        tvl = tvl.ffill()
        df["tvl_roc"] = tvl.pct_change(config.tvl_roc_window)
    else:
        df["tvl_roc"] = np.nan

    # --- Liquidity Composite Score (0~2) ---
    # Each indicator: +1 if growing, 0 if NaN or flat/declining
    stab_roc: pd.Series = df["stablecoin_roc"]  # type: ignore[assignment]
    tvl_roc: pd.Series = df["tvl_roc"]  # type: ignore[assignment]
    stab_score = pd.Series(
        np.where(stab_roc > 0, 1.0, 0.0),
        index=df.index,
    )
    tvl_score = pd.Series(
        np.where(tvl_roc > 0, 1.0, 0.0),
        index=df.index,
    )
    # NaN → 0 (neutral)
    stab_score = stab_score.where(stab_roc.notna(), 0.0)
    tvl_score = tvl_score.where(tvl_roc.notna(), 0.0)
    df["liq_score"] = stab_score + tvl_score

    # --- Fear & Greed (optional, graceful degradation) ---
    has_fg = "oc_fear_greed" in df.columns
    if has_fg:
        fg: pd.Series = df["oc_fear_greed"]  # type: ignore[assignment]
        fg = fg.ffill()
        df["oc_fear_greed"] = fg
        df["fg_ma"] = sma(fg, period=config.fg_ma_window)
    else:
        df["oc_fear_greed"] = np.nan
        df["fg_ma"] = np.nan

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
