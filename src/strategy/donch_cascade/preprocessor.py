"""Donchian Cascade MTF 전처리 모듈.

12H-equivalent Donchian Channel (lookback x htf_multiplier) +
4H EMA confirmation + 변동성 스케일러를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.market.indicators import (
    donchian_channel,
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.donch_cascade.config import DonchCascadeConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: DonchCascadeConfig) -> pd.DataFrame:
    """Donchian Cascade MTF feature 계산.

    Calculated Columns:
        - dc_upper_{lb}, dc_lower_{lb}: 3-scale Donchian Channel (actual lookback)
        - confirm_ema: 4H EMA (진입 확인용)
        - returns: log return
        - realized_vol: 연환산 실현 변동성
        - vol_scalar: 변동성 스케일러
        - drawdown: HEDGE_ONLY용 drawdown

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수, 4H 타임프레임)
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

    # --- 3-Scale Donchian Channels (12H-equivalent) ---
    actual_lookbacks = config.actual_lookbacks()
    for lb in actual_lookbacks:
        upper, _mid, lower = donchian_channel(high, low, lb)
        df[f"dc_upper_{lb}"] = upper
        df[f"dc_lower_{lb}"] = lower

    # --- 4H Confirmation EMA ---
    df["confirm_ema"] = close.ewm(span=config.confirm_ema_period, adjust=False).mean()

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
