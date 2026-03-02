"""VWAP-Channel Multi-Scale 전처리 모듈.

OHLCV 데이터에서 3-scale VWAP channel feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).

Rolling VWAP = sum(typical_price * volume, window) / sum(volume, window)
Band = VWAP +/- multiplier * ATR
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.market.indicators import (
    atr,
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.vwap_channel_12h.config import VwapChannelConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def _rolling_vwap(
    typical_price: pd.Series,
    volume: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling VWAP 계산.

    VWAP = sum(typical_price * volume, window) / sum(volume, window)

    Args:
        typical_price: (high + low + close) / 3
        volume: 거래량 시리즈
        window: rolling lookback 기간

    Returns:
        Rolling VWAP Series
    """
    tp_vol: pd.Series = typical_price * volume  # type: ignore[assignment]
    sum_tp_vol = tp_vol.rolling(window=window, min_periods=window).sum()
    sum_vol = volume.rolling(window=window, min_periods=window).sum()
    # volume 합이 0인 경우 방어 (극히 드물지만 안전장치)
    return sum_tp_vol / sum_vol.clip(lower=1e-10)  # type: ignore[return-value]


def preprocess(df: pd.DataFrame, config: VwapChannelConfig) -> pd.DataFrame:
    """VWAP-Channel Multi-Scale feature 계산.

    Calculated Columns:
        - vwap_{s}: 3-scale Rolling VWAP
        - vwap_upper_{s}: VWAP + multiplier * ATR
        - vwap_lower_{s}: VWAP - multiplier * ATR
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
    volume: pd.Series = df["volume"]  # type: ignore[assignment]

    # --- Typical Price ---
    typical_price: pd.Series = (high + low + close) / 3  # type: ignore[assignment]

    # --- ATR (band width 계산용) ---
    atr_val = atr(high, low, close, period=config.atr_period)

    scales = (config.scale_short, config.scale_mid, config.scale_long)

    # --- 3-Scale VWAP Channels ---
    for s in scales:
        vwap_s = _rolling_vwap(typical_price, volume, window=s)
        df[f"vwap_{s}"] = vwap_s
        df[f"vwap_upper_{s}"] = vwap_s + config.band_multiplier * atr_val
        df[f"vwap_lower_{s}"] = vwap_s - config.band_multiplier * atr_val

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
