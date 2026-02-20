"""EMA Ribbon Momentum 전처리 — 리본 정렬도 + ROC 계산."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.market.indicators import (
    drawdown,
    ema,
    log_returns,
    realized_volatility,
    roc,
    volatility_scalar,
)

if TYPE_CHECKING:
    from src.strategy.ema_ribbon_mom.config import EmaRibbonMomConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def _compute_alignment(ema_values: list[pd.Series]) -> pd.Series:
    """EMA 리본의 정렬도를 계산한다.

    N개 EMA의 모든 인접 쌍이 올바른 순서인지 확인.
    Bullish alignment: EMA[0] > EMA[1] > ... > EMA[N-1] (짧은 기간이 위)
    Bearish alignment: EMA[0] < EMA[1] < ... < EMA[N-1] (짧은 기간이 아래)

    Returns:
        정렬도 (-1 ~ +1). +1=완전 bullish, -1=완전 bearish, 0=혼합.
    """
    n = len(ema_values)
    total_pairs = n - 1

    # 인접 쌍 비교: EMA[i] > EMA[i+1] → bullish pair
    bull_count = pd.Series(np.zeros(len(ema_values[0])), index=ema_values[0].index)
    bear_count = pd.Series(np.zeros(len(ema_values[0])), index=ema_values[0].index)

    for i in range(total_pairs):
        bull_count += (ema_values[i] > ema_values[i + 1]).astype(float)
        bear_count += (ema_values[i] < ema_values[i + 1]).astype(float)

    # 정규화: (bull - bear) / total_pairs → -1 ~ +1
    alignment: pd.Series = (bull_count - bear_count) / total_pairs
    return alignment


def preprocess(df: pd.DataFrame, config: EmaRibbonMomConfig) -> pd.DataFrame:
    """EMA 리본 정렬도 + ROC 모멘텀 지표를 계산한다.

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

    # 피보나치 EMA 리본
    ema_values: list[pd.Series] = []
    for period in config.ema_periods:
        ema_val = ema(close, span=period)
        df[f"ema_{period}"] = ema_val
        ema_values.append(ema_val)

    # 리본 정렬도 (-1 ~ +1)
    df["alignment"] = _compute_alignment(ema_values)

    # ROC 모멘텀
    df["roc"] = roc(close, period=config.roc_period)

    # 방향: 최단 EMA vs 최장 EMA
    df["ribbon_direction"] = np.sign(ema_values[0] - ema_values[-1])

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

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
