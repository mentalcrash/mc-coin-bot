"""Drawdown-Recovery Phase 전처리 모듈.

OHLCV 데이터에서 drawdown depth, recovery ratio, recovery phase 상태를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

import numpy as np
import pandas as pd

from src.market.indicators import (
    atr,
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)
from src.strategy.dd_recovery_phase.config import DDRecoveryPhaseConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: DDRecoveryPhaseConfig) -> pd.DataFrame:
    """Drawdown-Recovery Phase feature 계산.

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

    # --- Drawdown ---
    dd = drawdown(close)
    df["drawdown"] = dd

    # --- Rolling max for recovery calculation ---
    rolling_max: pd.Series = close.rolling(  # type: ignore[assignment]
        window=config.dd_lookback,
        min_periods=config.dd_lookback,
    ).max()
    df["rolling_max"] = rolling_max

    # --- Recovery ratio: how much of the drawdown has been recovered ---
    # dd_depth = (close - rolling_max) / rolling_max  (same as drawdown, always <= 0)
    # trough = rolling minimum of close since peak
    # recovery_ratio = (close - trough) / (rolling_max - trough) when trough < rolling_max
    rolling_min: pd.Series = close.rolling(  # type: ignore[assignment]
        window=config.dd_lookback,
        min_periods=config.dd_lookback,
    ).min()
    range_val = rolling_max - rolling_min
    range_safe = range_val.clip(lower=1e-10)
    recovery_from_trough: pd.Series = (close - rolling_min) / range_safe  # type: ignore[assignment]
    df["recovery_ratio_val"] = recovery_from_trough

    # --- Was in deep drawdown recently ---
    # Track the minimum drawdown in the lookback window
    rolling_min_dd: pd.Series = dd.rolling(  # type: ignore[assignment]
        window=config.dd_lookback,
        min_periods=1,
    ).min()
    df["worst_dd"] = rolling_min_dd
    df["was_deep_dd"] = (rolling_min_dd <= config.dd_threshold).astype(int)

    # --- Momentum confirmation ---
    mom_return = close / close.shift(config.momentum_lookback) - 1.0
    df["momentum"] = np.sign(mom_return)

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    return df
