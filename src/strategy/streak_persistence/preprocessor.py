"""Return Streak Persistence 전처리 모듈.

OHLCV 데이터에서 연속 양봉/음봉 streak, momentum 방향을 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

import numpy as np
import pandas as pd

from src.market.indicators import (
    atr,
    count_consecutive,
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)
from src.strategy.streak_persistence.config import StreakPersistenceConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: StreakPersistenceConfig) -> pd.DataFrame:
    """Return Streak Persistence feature 계산.

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
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

    # --- Simple returns for streak detection ---
    simple_ret: pd.Series = close.pct_change()  # type: ignore[assignment]

    # --- Positive streak: consecutive positive return bars ---
    positive_mask = (simple_ret > 0).to_numpy().astype(bool)
    # First bar has NaN return → treat as False
    positive_mask[0] = False
    df["positive_streak"] = pd.Series(
        count_consecutive(positive_mask),
        index=df.index,
    )

    # --- Negative streak: consecutive negative return bars ---
    negative_mask = (simple_ret < 0).to_numpy().astype(bool)
    negative_mask[0] = False
    df["negative_streak"] = pd.Series(
        count_consecutive(negative_mask),
        index=df.index,
    )

    # --- Momentum direction ---
    mom_return = close / close.shift(config.momentum_lookback) - 1.0
    df["momentum"] = np.sign(mom_return)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    return df
