"""EMA Multi-Cross 전처리 — 3쌍 EMA 크로스 지표 계산."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.market.indicators import (
    drawdown,
    ema_cross,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.ema_multi_cross.config import EmaMultiCrossConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: EmaMultiCrossConfig) -> pd.DataFrame:
    """3쌍 EMA Cross 지표를 계산한다.

    Args:
        df: OHLCV DataFrame.
        config: 전략 설정.

    Returns:
        지표가 추가된 DataFrame (원본 불변).

    Raises:
        ValueError: 필수 컬럼 누락 시.
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    df = df.copy()

    close: pd.Series = df["close"]  # type: ignore[assignment]

    # --- 3쌍 EMA Cross ---
    df["cross1"] = ema_cross(close, fast=config.pair1_fast, slow=config.pair1_slow)
    df["cross2"] = ema_cross(close, fast=config.pair2_fast, slow=config.pair2_slow)
    df["cross3"] = ema_cross(close, fast=config.pair3_fast, slow=config.pair3_slow)

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
