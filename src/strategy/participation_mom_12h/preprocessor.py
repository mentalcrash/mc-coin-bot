"""Participation Momentum 전처리 모듈.

12H OHLCV + Trade Flow(tflow_intensity) 데이터에서 feature를 계산.
tflow_intensity 컬럼은 optional (Graceful Degradation).
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.participation_mom_12h.config import ParticipationMomConfig

from src.market.indicators import (
    drawdown,
    ema,
    log_returns,
    realized_volatility,
    rolling_zscore,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})

# Trade Flow 컬럼 (optional -- Graceful Degradation)
_TFLOW_INTENSITY = "tflow_intensity"


def preprocess(df: pd.DataFrame, config: ParticipationMomConfig) -> pd.DataFrame:
    """Participation Momentum feature 계산.

    Calculated columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - intensity_zscore: tflow_intensity의 rolling Z-score (부재 시 0)
        - mom_direction: EMA fast/slow 기반 모멘텀 방향 (+1/-1)
        - mom_strength: 모멘텀 수익률 (lookback)
        - drawdown: rolling drawdown (HEDGE_ONLY용)

    Args:
        df: OHLCV (+ tflow optional) DataFrame (DatetimeIndex 필수)
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

    # === Trade Flow Context (Graceful Degradation) ===

    # --- Intensity Z-score ---
    if _TFLOW_INTENSITY in df.columns:
        intensity: pd.Series = df[_TFLOW_INTENSITY].ffill()  # type: ignore[assignment]
        df["intensity_zscore"] = rolling_zscore(intensity, window=config.intensity_zscore_window)
    else:
        df["intensity_zscore"] = 0.0  # 중립 (Graceful Degradation)

    # --- Momentum Direction (EMA Cross) ---
    ema_fast = ema(close, span=config.mom_ema_fast)
    ema_slow = ema(close, span=config.mom_ema_slow)
    df["mom_direction"] = pd.Series(
        np.where(ema_fast > ema_slow, 1, -1),
        index=df.index,
        dtype=int,
    )

    # --- Momentum Strength (lookback return) ---
    df["mom_strength"] = close / close.shift(config.mom_lookback) - 1.0

    # --- Drawdown (HEDGE_ONLY/FULL 용) ---
    df["drawdown"] = drawdown(close)

    return df
