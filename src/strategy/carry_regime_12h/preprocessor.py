"""Carry-Regime Trend 전처리 모듈.

OHLCV + funding_rate 데이터에서 multi-scale EMA trend 및
FR percentile feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.carry_regime_12h.config import CarryRegimeConfig

from src.market.indicators import (
    atr,
    drawdown,
    ema,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: CarryRegimeConfig) -> pd.DataFrame:
    """Carry-Regime Trend feature 계산.

    Args:
        df: OHLCV + optional funding_rate DataFrame (DatetimeIndex 필수)
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

    # --- Multi-Scale EMA ---
    ema_fast: pd.Series = ema(close, span=config.ema_fast)
    ema_mid: pd.Series = ema(close, span=config.ema_mid)
    ema_slow: pd.Series = ema(close, span=config.ema_slow)
    df["ema_fast"] = ema_fast
    df["ema_mid"] = ema_mid
    df["ema_slow"] = ema_slow

    # --- EMA Alignment Score ---
    # +1 for each correct ordering: fast>mid, mid>slow, close>fast (long)
    # -1 for each reverse ordering (short)
    # Score range: [-3, +3], normalized to [-1, +1]
    score = np.sign(ema_fast - ema_mid) + np.sign(ema_mid - ema_slow) + np.sign(close - ema_fast)
    df["ema_alignment"] = score / 3.0

    # --- FR Percentile (optional, graceful degradation) ---
    if "funding_rate" in df.columns:
        fr: pd.Series = df["funding_rate"].ffill()  # type: ignore[assignment]
        # Absolute FR percentile over rolling window
        abs_fr = fr.abs()
        df["fr_percentile"] = abs_fr.rolling(
            window=config.fr_percentile_window,
            min_periods=max(config.fr_percentile_window // 2, 1),
        ).rank(pct=True)
        df["funding_rate_clean"] = fr
    else:
        # Graceful degradation: no FR data → neutral percentile (0.5)
        df["fr_percentile"] = 0.5
        df["funding_rate_clean"] = 0.0

    # --- Drawdown (HEDGE_ONLY / exit 참조용) ---
    df["drawdown"] = drawdown(close)

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    return df
