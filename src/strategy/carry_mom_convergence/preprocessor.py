"""Carry-Momentum Convergence 전처리 모듈 (Derivatives).

OHLCV + funding_rate 데이터에서 전략 feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.carry_mom_convergence.config import CarryMomConvergenceConfig

from src.market.indicators import (
    drawdown,
    ema,
    log_returns,
    realized_volatility,
    rolling_return,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "funding_rate"})


def preprocess(df: pd.DataFrame, config: CarryMomConvergenceConfig) -> pd.DataFrame:
    """Carry-Momentum Convergence feature 계산.

    Calculated columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - price_mom: rolling return (lookback)
        - ema_fast / ema_slow: EMA cross 추세 방향
        - trend_direction: EMA cross 기반 추세 (+1/-1)
        - avg_funding_rate: rolling mean funding rate
        - fr_zscore: funding rate z-score
        - convergence_score: 가격-FR 수렴/발산 점수
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
    df["price_mom"] = rolling_return(close, period=config.mom_lookback)

    # --- EMA Cross Trend Direction ---
    fast = ema(close, span=config.mom_fast)
    slow = ema(close, span=config.mom_slow)
    df["ema_fast"] = fast
    df["ema_slow"] = slow
    df["trend_direction"] = np.where(fast > slow, 1, np.where(fast < slow, -1, 0))

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

    # --- Convergence Score ---
    # FR의 방향 (양수 FR → 롱 과밀 → 가격 하락 압력 = -1, 음수 FR → 숏 과밀 → +1)
    fr_direction = np.where(avg_fr > 0, -1, np.where(avg_fr < 0, 1, 0))
    # 가격 추세와 FR implied direction이 같으면 convergence
    trend_dir = df["trend_direction"].to_numpy()
    convergence = np.where(
        (trend_dir == fr_direction) & (trend_dir != 0),
        config.convergence_boost,
        np.where(
            (trend_dir != 0) & (fr_direction != 0) & (trend_dir != fr_direction),
            config.divergence_penalty,
            1.0,  # neutral (FR near zero or trend unclear)
        ),
    )
    df["convergence_score"] = convergence

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
