"""EMA Cross Base 전처리 — 지표 계산 (벡터화)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.market.indicators import (
    ema,
    ema_cross,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.ema_cross_base.config import EmaCrossBaseConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: EmaCrossBaseConfig) -> pd.DataFrame:
    """EMA Cross 지표를 계산한다.

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

    # EMA
    df["ema_fast"] = ema(close, span=config.fast_period)
    df["ema_slow"] = ema(close, span=config.slow_period)

    # EMA Cross signal (fast/slow - 1)
    df["ema_cross"] = ema_cross(close, fast=config.fast_period, slow=config.slow_period)

    # Volatility scalar
    returns = log_returns(close)
    df["returns"] = returns

    realized_vol = realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    df["realized_vol"] = realized_vol

    df["vol_scalar"] = volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    return df
