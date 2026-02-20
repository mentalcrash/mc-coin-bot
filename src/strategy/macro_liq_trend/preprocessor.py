"""Macro-Liquidity Adaptive Trend 전처리 모듈.

OHLCV + Macro 데이터(DXY, VIX, SPY, Stablecoin)에서 전략 feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

import pandas as pd

from src.market.indicators import (
    atr,
    log_returns,
    realized_volatility,
    volatility_scalar,
)
from src.market.indicators.composite import drawdown, rolling_zscore
from src.market.indicators.trend import sma
from src.strategy.macro_liq_trend.config import MacroLiqTrendConfig

_REQUIRED_COLUMNS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "macro_dxy",
        "macro_vix",
        "macro_spy",
        "oc_stablecoin_total_circulating_usd",
    }
)


def preprocess(df: pd.DataFrame, config: MacroLiqTrendConfig) -> pd.DataFrame:
    """Macro-Liquidity Adaptive Trend feature 계산.

    Args:
        df: OHLCV + Macro DataFrame (DatetimeIndex 필수)
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

    # --- Macro Features (forward-fill for daily macro in 1D bars) ---
    dxy: pd.Series = df["macro_dxy"].ffill()  # type: ignore[assignment]
    vix: pd.Series = df["macro_vix"].ffill()  # type: ignore[assignment]
    spy: pd.Series = df["macro_spy"].ffill()  # type: ignore[assignment]
    stab: pd.Series = df["oc_stablecoin_total_circulating_usd"].ffill()  # type: ignore[assignment]

    # --- DXY ROC (inverted: DXY down = liquidity up) ---
    dxy_roc = dxy.pct_change(config.dxy_roc_period)
    df["dxy_roc"] = dxy_roc
    df["dxy_z"] = rolling_zscore(-dxy_roc, window=config.zscore_window)

    # --- VIX ROC (inverted: VIX down = risk-on) ---
    vix_roc = vix.pct_change(config.vix_roc_period)
    df["vix_roc"] = vix_roc
    df["vix_z"] = rolling_zscore(-vix_roc, window=config.zscore_window)

    # --- SPY ROC (direct: SPY up = risk-on) ---
    spy_roc = spy.pct_change(config.spy_roc_period)
    df["spy_roc"] = spy_roc
    df["spy_z"] = rolling_zscore(spy_roc, window=config.zscore_window)

    # --- Stablecoin change rate (direct: supply up = liquidity inflow) ---
    stab_change = stab.pct_change(config.stab_change_period)
    df["stab_change"] = stab_change
    df["stab_z"] = rolling_zscore(stab_change, window=config.zscore_window)

    # --- Composite Macro Liquidity Score (equal-weight average of z-scores) ---
    df["macro_liq_score"] = (df["dxy_z"] + df["vix_z"] + df["spy_z"] + df["stab_z"]) / 4.0

    # --- Price Momentum (SMA cross) ---
    df["sma_price"] = sma(close, period=config.price_mom_period)

    # --- Drawdown (HEDGE_ONLY 용) ---
    df["drawdown"] = drawdown(close)

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    return df
