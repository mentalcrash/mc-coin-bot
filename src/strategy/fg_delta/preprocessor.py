"""Fear & Greed Delta 전처리 모듈.

OHLCV + on-chain(oc_fear_greed) 데이터에서 F&G 변화율 feature 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.fg_delta.config import FgDeltaConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "oc_fear_greed"})


def preprocess(df: pd.DataFrame, config: FgDeltaConfig) -> pd.DataFrame:
    """Fear & Greed Delta feature 계산.

    Args:
        df: OHLCV + oc_fear_greed DataFrame
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

    # --- Fear & Greed Delta ---
    fg: pd.Series = df["oc_fear_greed"]  # type: ignore[assignment]
    fg = fg.ffill()  # merge_asof 후 NaN 처리

    # Delta: 현재 F&G - N일 전 F&G
    fg_delta = fg - fg.shift(config.fg_delta_window)
    df["fg_delta"] = fg_delta

    # Smoothed delta
    df["fg_delta_smooth"] = fg_delta.rolling(
        window=config.fg_smooth_window, min_periods=config.fg_smooth_window
    ).mean()

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
