"""Multi-Scale Volatility Ratio 전처리 모듈.

OHLCV에서 multi-scale vol ratio + momentum features 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.ms_vol_ratio.config import MSVolRatioConfig

from src.market.indicators import (
    atr,
    drawdown,
    ema,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: MSVolRatioConfig) -> pd.DataFrame:
    """Multi-Scale Volatility Ratio feature 계산.

    Args:
        df: OHLCV DataFrame
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

    # --- Short-Term Volatility ---
    short_vol = returns.rolling(
        window=config.short_vol_window,
        min_periods=config.short_vol_window,
    ).std()
    df["short_vol"] = short_vol

    # --- Long-Term Volatility ---
    long_vol = returns.rolling(
        window=config.long_vol_window,
        min_periods=config.long_vol_window,
    ).std()
    df["long_vol"] = long_vol

    # --- Vol Ratio (smoothed) ---
    raw_ratio = short_vol / long_vol.clip(lower=1e-10)
    df["vol_ratio"] = ema(raw_ratio, span=config.ratio_smooth)

    # --- Momentum Return ---
    df["mom_return"] = returns.rolling(
        window=config.mom_lookback,
        min_periods=config.mom_lookback,
    ).sum()

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    # --- Drawdown (HEDGE_ONLY) ---
    df["drawdown"] = drawdown(close)

    return df
