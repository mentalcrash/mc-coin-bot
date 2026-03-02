"""MAD-Channel Multi-Scale Trend 전처리 모듈.

3종 채널(Donchian/Keltner/MAD) x 3스케일(20/60/150) feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.market.indicators import (
    donchian_channel,
    drawdown,
    keltner_channels,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.mad_channel_trend.config import MadChannelTrendConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: MadChannelTrendConfig) -> pd.DataFrame:
    """MAD-Channel Multi-Scale Trend feature 계산.

    Calculated Columns:
        - dc_upper_{s}, dc_lower_{s}: 3-scale Donchian Channel
        - kc_upper_{s}, kc_lower_{s}: 3-scale Keltner Channels
        - mad_upper_{s}, mad_lower_{s}: 3-scale Median-MAD Channel (L1-robust)
        - returns: log return
        - realized_vol: 연환산 실현 변동성
        - vol_scalar: 변동성 스케일러
        - drawdown: HEDGE_ONLY용 drawdown

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
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

    scales = (config.scale_short, config.scale_mid, config.scale_long)

    # --- 3-Scale Donchian Channels ---
    for s in scales:
        upper, _mid, lower = donchian_channel(high, low, s)
        df[f"dc_upper_{s}"] = upper
        df[f"dc_lower_{s}"] = lower

    # --- 3-Scale Keltner Channels ---
    for s in scales:
        kc_upper, _kc_mid, kc_lower = keltner_channels(
            high,
            low,
            close,
            ema_period=s,
            atr_period=s,
            multiplier=config.keltner_multiplier,
        )
        df[f"kc_upper_{s}"] = kc_upper
        df[f"kc_lower_{s}"] = kc_lower

    # --- 3-Scale Median-MAD Channels (L1-robust) ---
    for s in scales:
        median = close.rolling(s).median()
        mad = close.rolling(s).apply(
            lambda x: np.median(np.abs(x - np.median(x))),  # type: ignore[arg-type]
            raw=True,
        )
        scaled_mad = config.mad_multiplier * 1.4826 * mad
        df[f"mad_upper_{s}"] = median + scaled_mad
        df[f"mad_lower_{s}"] = median - scaled_mad

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

    return df
