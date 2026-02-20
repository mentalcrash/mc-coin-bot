"""Disposition Breakout 전처리 모듈.

Rolling high 대비 가격 위치 + 모멘텀 feature를 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.disp_breakout.config import DispBreakoutConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    roc,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: DispBreakoutConfig) -> pd.DataFrame:
    """Disposition Breakout feature 계산.

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

    # --- Rolling High ---
    rolling_high = high.rolling(window=config.high_window, min_periods=config.high_window).max()
    df["rolling_high"] = rolling_high

    # --- Price relative to rolling high ---
    safe_rolling_high = rolling_high.clip(lower=1e-10)
    df["price_to_high"] = close / safe_rolling_high

    # --- Momentum ---
    df["momentum"] = roc(close, period=config.momentum_window)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
