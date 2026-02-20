"""F&G EMA Long-Cycle 전처리 모듈.

OHLCV + on-chain(oc_fear_greed) 데이터에서 장기 EMA 크로스오버 feature 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.fg_ema_cycle.config import FgEmaCycleConfig

from src.market.indicators import (
    drawdown,
    ema,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "oc_fear_greed"})


def preprocess(df: pd.DataFrame, config: FgEmaCycleConfig) -> pd.DataFrame:
    """F&G EMA Long-Cycle feature 계산.

    Args:
        df: OHLCV + oc_fear_greed DataFrame (DatetimeIndex 필수)
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

    # --- Fear & Greed EMA ---
    fg: pd.Series = df["oc_fear_greed"]  # type: ignore[assignment]
    fg = fg.ffill()
    df["oc_fear_greed"] = fg

    # 장기 EMA (24주 ≈ 168일)
    df["fg_ema_slow"] = ema(fg, span=config.ema_slow_span)

    # 단기 EMA (6주 ≈ 42일)
    df["fg_ema_fast"] = ema(fg, span=config.ema_fast_span)

    # --- Cycle Position: 장기 EMA의 50 대비 편차 (strength 계산용) ---
    df["cycle_position"] = (df["fg_ema_slow"] - 50.0) / 50.0

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
