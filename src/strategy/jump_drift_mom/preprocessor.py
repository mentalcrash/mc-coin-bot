"""Jump Drift Momentum 전처리 모듈.

OHLCV 데이터에서 bipower variation 기반 jump component를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.jump_drift_mom.config import JumpDriftMomConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    rolling_return,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})

# Bipower variation scaling constant: mu_1^2 = (sqrt(2/pi))^2
_MU1_SQ = 2.0 / np.pi


def preprocess(df: pd.DataFrame, config: JumpDriftMomConfig) -> pd.DataFrame:
    """Jump Drift Momentum feature 계산.

    Calculated columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - realized_var: Rolling realized variance (sum of squared returns)
        - bipower_var: Rolling bipower variation (sum of |r_t| * |r_{t-1}|)
        - jump_var: Jump variance = max(0, RV - BPV)
        - jump_ratio: Jump variance / realized variance
        - jump_zscore: jump_ratio의 rolling z-score
        - drift_direction: Post-jump drift 방향 (가격 변화 부호)
        - drawdown: rolling drawdown (HEDGE_ONLY용)

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

    # --- Realized Variance ---
    returns_sq = returns**2
    rv: pd.Series = returns_sq.rolling(  # type: ignore[assignment]
        window=config.rv_window, min_periods=config.rv_window
    ).sum()
    df["realized_var"] = rv

    # --- Bipower Variation ---
    # BPV = (pi/2) * sum(|r_t| * |r_{t-1}|) over window
    abs_ret = returns.abs()
    abs_ret_lag = abs_ret.shift(1)
    bpv_products = abs_ret * abs_ret_lag
    bpv_raw: pd.Series = bpv_products.rolling(  # type: ignore[assignment]
        window=config.bpv_window, min_periods=config.bpv_window
    ).sum()
    # Scale by pi/2 to get consistent estimator of integrated variance
    bpv = bpv_raw / _MU1_SQ
    df["bipower_var"] = bpv

    # --- Jump Variance ---
    # Jump component = max(0, RV - BPV)
    jump_var = (rv - bpv).clip(lower=0)
    df["jump_var"] = jump_var

    # --- Jump Ratio ---
    jump_ratio = jump_var / rv.clip(lower=1e-20)
    df["jump_ratio"] = jump_ratio

    # --- Jump Z-Score ---
    jr_mean: pd.Series = jump_ratio.rolling(  # type: ignore[assignment]
        window=config.rv_window, min_periods=config.rv_window
    ).mean()
    jr_std: pd.Series = jump_ratio.rolling(  # type: ignore[assignment]
        window=config.rv_window, min_periods=config.rv_window
    ).std()
    df["jump_zscore"] = (jump_ratio - jr_mean) / jr_std.clip(lower=1e-10)

    # --- Drift Direction ---
    # Post-jump drift 방향: 최근 drift_lookback의 cumulative return 부호
    df["drift_direction"] = pd.Series(
        np.sign(rolling_return(close, period=config.drift_lookback)),
        index=df.index,
    )

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
