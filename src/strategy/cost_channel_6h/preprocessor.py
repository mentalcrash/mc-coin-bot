"""Cost-Penalized Multi-Scale Channel 전처리 모듈.

3종 채널(Donchian/Keltner/BB) x 3스케일(15/45/120) feature + ATR 기대이익 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.market.indicators import (
    atr,
    bollinger_bands,
    donchian_channel,
    drawdown,
    keltner_channels,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.cost_channel_6h.config import CostChannel6hConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: CostChannel6hConfig) -> pd.DataFrame:
    """Cost-Penalized Multi-Scale Channel feature 계산.

    Calculated Columns:
        - dc_upper_{s}, dc_lower_{s}: 3-scale Donchian Channel
        - kc_upper_{s}, kc_lower_{s}: 3-scale Keltner Channels
        - bb_upper_{s}, bb_lower_{s}: 3-scale Bollinger Bands
        - returns: log return
        - realized_vol: 연환산 실현 변동성
        - vol_scalar: 변동성 스케일러
        - drawdown: HEDGE_ONLY용 drawdown
        - atr_profit: 기대 이익 추정용 ATR

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

    # --- 3-Scale Bollinger Bands ---
    for s in scales:
        bb_upper, _bb_mid, bb_lower = bollinger_bands(
            close,
            period=s,
            std_dev=config.bb_std_dev,
        )
        df[f"bb_upper_{s}"] = bb_upper
        df[f"bb_lower_{s}"] = bb_lower

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

    # --- ATR for Expected Profit Estimation ---
    df["atr_profit"] = atr(high, low, close, period=config.atr_profit_window)

    return df
