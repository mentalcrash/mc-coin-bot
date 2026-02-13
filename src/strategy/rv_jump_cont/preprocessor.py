"""RV-Jump Continuation 전처리 모듈.

OHLCV 데이터에서 realized variance, bipower variation, jump ratio를 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

import numpy as np
import pandas as pd

from src.strategy.rv_jump_cont.config import RvJumpContConfig
from src.strategy.tsmom.preprocessor import (
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})

# Bipower variation scaling constant: mu_1^2 = (sqrt(2/pi))^2 = 2/pi
_MU1_SQUARED = 2.0 / np.pi


def preprocess(df: pd.DataFrame, config: RvJumpContConfig) -> pd.DataFrame:
    """RV-Jump Continuation feature 계산.

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

    # --- Returns ---
    returns = calculate_returns(close)
    df["returns"] = returns

    # --- Realized Volatility ---
    realized_vol = calculate_realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    df["realized_vol"] = realized_vol

    # --- Vol Scalar ---
    df["vol_scalar"] = calculate_volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # --- Realized Variance: sum(r^2) over window ---
    rv = (returns**2).rolling(window=config.rv_window, min_periods=config.rv_window).sum()

    # --- Bipower Variation: (pi/2) * sum(|r_t| * |r_{t-1}|) over window ---
    abs_ret = returns.abs()
    abs_ret_lag = abs_ret.shift(1)
    bv_raw = (
        (abs_ret * abs_ret_lag).rolling(window=config.bv_window, min_periods=config.bv_window).sum()
    )
    bv = bv_raw / _MU1_SQUARED  # scale to comparable variance units

    # --- Jump Ratio: RV / BV (>1 means jump component present) ---
    bv_safe = bv.clip(lower=1e-20)
    df["jump_ratio"] = rv / bv_safe

    # --- Momentum direction for jump ---
    mom_return = close / close.shift(config.mom_lookback) - 1.0
    df["mom_direction"] = np.sign(mom_return)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = calculate_drawdown(close)

    return df
