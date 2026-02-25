"""Multi-Source Directional Composite 전처리 모듈.

3개 직교 데이터 소스에서 directional sub-signal feature를 계산한다.
Source 1: OHLCV momentum (EMA cross + rolling return)
Source 2: Stablecoin velocity proxy (volume/close ratio)
Source 3: Fear & Greed sentiment delta (oc_fear_greed)

모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.multi_source_composite.config import MultiSourceCompositeConfig

from src.market.indicators import (
    drawdown,
    ema,
    log_returns,
    realized_volatility,
    rolling_return,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "oc_fear_greed"})


def preprocess(df: pd.DataFrame, config: MultiSourceCompositeConfig) -> pd.DataFrame:
    """Multi-Source Directional Composite feature 계산.

    Calculated columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - ema_fast / ema_slow: EMA cross (Source 1)
        - price_mom: rolling return (Source 1)
        - mom_direction: EMA cross 기반 모멘텀 방향 (+1/0/-1)
        - velocity_fast / velocity_slow: stablecoin velocity proxy MA (Source 2)
        - velocity_direction: velocity cross 방향 (+1/0/-1)
        - fg_delta_smooth: F&G smoothed delta (Source 3)
        - fg_direction: F&G delta 방향 (+1/0/-1)
        - drawdown: rolling drawdown (HEDGE_ONLY용)

    Args:
        df: OHLCV + oc_fear_greed DataFrame (DatetimeIndex 필수)
        config: 전략 설정

    Returns:
        feature가 추가된 새 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    import numpy as np

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    df = df.copy()

    close: pd.Series = df["close"]  # type: ignore[assignment]
    volume: pd.Series = df["volume"]  # type: ignore[assignment]

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

    # =====================================================================
    # Source 1: OHLCV Momentum (EMA cross + rolling return confirmation)
    # =====================================================================
    fast = ema(close, span=config.mom_fast)
    slow = ema(close, span=config.mom_slow)
    df["ema_fast"] = fast
    df["ema_slow"] = slow

    price_mom = rolling_return(close, period=config.mom_lookback)
    df["price_mom"] = price_mom

    # Momentum direction: EMA cross 방향, rolling return으로 확인
    ema_dir = np.where(fast > slow, 1, np.where(fast < slow, -1, 0))
    mom_confirm = np.where(price_mom > 0, 1, np.where(price_mom < 0, -1, 0))
    # 두 지표가 합의할 때만 방향 부여
    df["mom_direction"] = np.where(ema_dir == mom_confirm, ema_dir, 0)

    # =====================================================================
    # Source 2: Stablecoin Velocity Proxy (volume/close ratio)
    # =====================================================================
    velocity_raw = volume / close.clip(lower=1e-10)

    velocity_fast: pd.Series = velocity_raw.rolling(  # type: ignore[assignment]
        window=config.velocity_fast_window,
        min_periods=config.velocity_fast_window,
    ).mean()
    velocity_slow: pd.Series = velocity_raw.rolling(  # type: ignore[assignment]
        window=config.velocity_slow_window,
        min_periods=config.velocity_slow_window,
    ).mean()
    df["velocity_fast"] = velocity_fast
    df["velocity_slow"] = velocity_slow

    # Velocity direction: fast > slow = liquidity increasing = bullish
    df["velocity_direction"] = np.where(
        velocity_fast > velocity_slow,
        1,
        np.where(velocity_fast < velocity_slow, -1, 0),
    )

    # =====================================================================
    # Source 3: Fear & Greed Sentiment Delta
    # =====================================================================
    fg: pd.Series = df["oc_fear_greed"]  # type: ignore[assignment]
    fg = fg.ffill()  # merge_asof 후 NaN 처리

    fg_delta = fg - fg.shift(config.fg_delta_window)
    fg_delta_smooth: pd.Series = fg_delta.rolling(  # type: ignore[assignment]
        window=config.fg_smooth_window,
        min_periods=config.fg_smooth_window,
    ).mean()
    df["fg_delta_smooth"] = fg_delta_smooth

    # F&G direction: delta > threshold = greed rising = bullish
    df["fg_direction"] = np.where(
        fg_delta_smooth > config.fg_threshold,
        1,
        np.where(fg_delta_smooth < -config.fg_threshold, -1, 0),
    )

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
