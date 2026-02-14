"""Funding Rate Carry (Vol-Conditioned) 전처리 모듈.

OHLCV + funding_rate 데이터에서 carry + vol conditioning features를 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.fr_carry_vol.config import FRCarryVolConfig

from src.market.indicators import (
    atr,
    drawdown,
    funding_rate_ma,
    funding_zscore,
    log_returns,
    realized_volatility,
    vol_percentile_rank,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "funding_rate"})


def preprocess(df: pd.DataFrame, config: FRCarryVolConfig) -> pd.DataFrame:
    """FR Carry Vol feature 계산.

    Args:
        df: OHLCV + funding_rate DataFrame
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
    funding_rate: pd.Series = df["funding_rate"]  # type: ignore[assignment]

    # NaN from merge_asof → ffill
    funding_rate = funding_rate.ffill()
    df["funding_rate"] = funding_rate

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

    # --- Funding Rate Features ---
    df["avg_funding_rate"] = funding_rate_ma(funding_rate, window=config.fr_lookback)
    df["funding_zscore"] = funding_zscore(
        funding_rate,
        ma_window=config.fr_lookback,
        zscore_window=config.fr_zscore_window,
    )

    # --- Vol Percentile Rank (conditioning) ---
    df["vol_pctile"] = vol_percentile_rank(
        realized_vol,
        window=config.vol_condition_window,
    )

    # --- Drawdown (HEDGE_ONLY) ---
    df["drawdown"] = drawdown(close)

    # --- ATR (trailing stop) ---
    df["atr"] = atr(high, low, close, period=14)

    return df
