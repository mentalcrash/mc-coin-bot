"""Basis-Momentum 전처리 모듈.

FR 변화율(1st derivative) z-score + 표준 vol-target feature를 계산.
funding_rate 컬럼 부재 시 0으로 대체 (Graceful Degradation — 시그널 없음, crash 없음).
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.basis_momentum.config import BasisMomentumConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})
_FR_COLUMN = "funding_rate"


def preprocess(df: pd.DataFrame, config: BasisMomentumConfig) -> pd.DataFrame:
    """Basis-Momentum feature 계산.

    Calculated Columns:
        - fr_change: FR N-period 변화 (1st derivative)
        - fr_change_std: FR 1-period diff의 rolling std (정규화 분모)
        - basis_mom: FR 모멘텀 z-score (fr_change / fr_change_std)
        - returns: log return
        - realized_vol: 연환산 실현 변동성
        - vol_scalar: 변동성 스케일러
        - drawdown: HEDGE_ONLY용 drawdown

    Args:
        df: OHLCV (+ funding_rate optional) DataFrame (DatetimeIndex 필수)
        config: 전략 설정

    Returns:
        feature가 추가된 새 DataFrame

    Raises:
        ValueError: 필수 OHLCV 컬럼 누락 시
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    df = df.copy()

    close: pd.Series = df["close"]  # type: ignore[assignment]

    # === Funding Rate Features (Graceful Degradation) ===

    # funding_rate 부재 시 0으로 대체 → basis_mom = 0 → 시그널 없음
    if _FR_COLUMN not in df.columns:
        df[_FR_COLUMN] = 0.0

    fr: pd.Series = df[_FR_COLUMN].fillna(0.0)  # type: ignore[assignment]

    # --- FR change (N-period 1st derivative) ---
    df["fr_change"] = fr.diff(config.fr_change_window)

    # --- FR change std (정규화 분모: 1-period diff의 rolling std) ---
    fr_diff: pd.Series = fr.diff(1)  # type: ignore[assignment]
    df["fr_change_std"] = fr_diff.rolling(config.fr_std_window).std()

    # --- Basis Momentum z-score ---
    fr_change_std: pd.Series = df["fr_change_std"].replace(0, np.nan)  # type: ignore[assignment]
    basis_mom: pd.Series = df["fr_change"] / fr_change_std  # type: ignore[assignment]
    df["basis_mom"] = basis_mom.fillna(0.0)

    # === Standard Vol-Target Features ===

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

    # --- Drawdown (HEDGE_ONLY/FULL 용) ---
    df["drawdown"] = drawdown(close)

    return df
