"""Donchian Filtered 전처리 모듈.

Donch-Multi 3-scale Donchian Channel + vol scalar에
funding rate z-score를 추가한다.
Graceful degradation: funding_rate 컬럼 없으면 fr_zscore=0.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.market.indicators import (
    donchian_channel,
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)
from src.market.indicators.derivatives import funding_zscore

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.donch_filtered.config import DonchFilteredConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: DonchFilteredConfig) -> pd.DataFrame:
    """Donchian Filtered feature 계산.

    Calculated Columns:
        - dc_upper_{lb}, dc_lower_{lb}: 3-scale Donchian Channel
        - returns: log return
        - realized_vol: 연환산 실현 변동성
        - vol_scalar: 변동성 스케일러
        - drawdown: HEDGE_ONLY용 drawdown
        - fr_zscore: funding rate z-score (derivatives 없으면 0.0)

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수).
            Optional: funding_rate 컬럼 (derivatives 데이터).
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

    # --- 3-Scale Donchian Channels ---
    for lb in (config.lookback_short, config.lookback_mid, config.lookback_long):
        upper, _mid, lower = donchian_channel(high, low, lb)
        df[f"dc_upper_{lb}"] = upper
        df[f"dc_lower_{lb}"] = lower

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

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    # --- Funding Rate Z-Score (Graceful Degradation) ---
    has_derivatives = "funding_rate" in df.columns
    if has_derivatives:
        fr_series: pd.Series = df["funding_rate"]  # type: ignore[assignment]
        df["fr_zscore"] = funding_zscore(
            fr_series,
            ma_window=config.fr_ma_window,
            zscore_window=config.fr_zscore_window,
        )
    else:
        df["fr_zscore"] = 0.0

    return df
