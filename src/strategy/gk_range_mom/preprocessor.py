"""GK Range Momentum 전처리 모듈.

OHLCV 데이터에서 close-in-range position + GK volatility feature 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.gk_range_mom.config import GkRangeMomConfig

from src.market.indicators import (
    drawdown,
    garman_klass_volatility,
    log_returns,
    realized_volatility,
    roc,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: GkRangeMomConfig) -> pd.DataFrame:
    """GK Range Momentum feature 계산.

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
    open_: pd.Series = df["open"]  # type: ignore[assignment]

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

    # --- Close-in-Range Position ---
    # (close - low) / (high - low) => 0~1 범위, 1에 가까우면 고점 마감
    hl_range = (high - low).clip(lower=1e-10)
    raw_range_pos = (close - low) / hl_range
    df["range_position"] = raw_range_pos.rolling(
        window=config.range_window, min_periods=config.range_window
    ).mean()

    # --- GK Volatility (bar-level) ---
    gk_vol = garman_klass_volatility(open_, high, low, close)
    df["gk_vol"] = gk_vol.rolling(window=config.gk_window, min_periods=config.gk_window).mean()

    # --- Momentum (ROC) ---
    df["momentum"] = roc(close, period=config.momentum_window)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
