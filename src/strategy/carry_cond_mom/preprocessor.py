"""Carry-Conditional Momentum 전처리 모듈 (Derivatives).

OHLCV + funding_rate 데이터에서 가격 모멘텀과 FR level agreement feature를 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.carry_cond_mom.config import CarryCondMomConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    rolling_return,
    volatility_scalar,
)


_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "funding_rate"})


def preprocess(df: pd.DataFrame, config: CarryCondMomConfig) -> pd.DataFrame:
    """Carry-Conditional Momentum feature 계산.

    Calculated columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - price_mom: 가격 모멘텀 (rolling return)
        - avg_funding_rate: rolling mean funding rate
        - fr_zscore: funding rate z-score
        - mom_direction: 가격 모멘텀 방향 부호 (+1/0/-1)
        - fr_direction: FR z-score 방향 부호 (+1/0/-1)
        - agreement: mom_direction * fr_direction (동의=+1, 불일치=-1)
        - drawdown: rolling drawdown (HEDGE_ONLY용)

    Args:
        df: OHLCV + funding_rate DataFrame (DatetimeIndex 필수)
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

    # --- Price Momentum ---
    price_mom = rolling_return(close, window=config.mom_lookback)
    df["price_mom"] = price_mom

    # --- Funding Rate Features ---
    funding_rate: pd.Series = df["funding_rate"]  # type: ignore[assignment]
    funding_rate = funding_rate.ffill()  # merge_asof 후 NaN 처리

    # Rolling mean FR
    avg_fr: pd.Series = funding_rate.rolling(  # type: ignore[assignment]
        window=config.fr_lookback, min_periods=config.fr_lookback
    ).mean()
    df["avg_funding_rate"] = avg_fr

    # FR z-score
    rolling_mean: pd.Series = avg_fr.rolling(  # type: ignore[assignment]
        window=config.fr_zscore_window, min_periods=config.fr_zscore_window
    ).mean()
    rolling_std: pd.Series = avg_fr.rolling(  # type: ignore[assignment]
        window=config.fr_zscore_window, min_periods=config.fr_zscore_window
    ).std()
    df["fr_zscore"] = (avg_fr - rolling_mean) / rolling_std.clip(lower=1e-10)

    # --- Direction Signs ---
    df["mom_direction"] = pd.Series(np.sign(price_mom), index=df.index)
    fr_zscore: pd.Series = df["fr_zscore"]  # type: ignore[assignment]
    df["fr_direction"] = pd.Series(np.sign(fr_zscore), index=df.index)

    # --- Agreement Score ---
    # +1: 동의 (both positive or both negative)
    # -1: 불일치 (opposite directions)
    # 0: 어느 한쪽이 0
    mom_dir: pd.Series = df["mom_direction"]  # type: ignore[assignment]
    fr_dir: pd.Series = df["fr_direction"]  # type: ignore[assignment]
    df["agreement"] = mom_dir * fr_dir

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
