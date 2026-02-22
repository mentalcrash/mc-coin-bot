"""Stablecoin Velocity Regime 전처리 모듈.

OHLCV 데이터에서 volume-based velocity proxy, z-score를 계산한다.
실제 on-chain stablecoin 데이터 대신 volume/close ratio로 velocity를 프록시.
모든 연산은 벡터화 (for 루프 금지).
"""

import pandas as pd

from src.market.indicators import (
    atr,
    drawdown,
    log_returns,
    realized_volatility,
    rolling_zscore,
    volatility_scalar,
)
from src.strategy.stablecoin_velocity.config import StablecoinVelocityConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: StablecoinVelocityConfig) -> pd.DataFrame:
    """Stablecoin Velocity Regime feature 계산.

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

    # --- Velocity Proxy: volume / close (turnover rate proxy) ---
    # Higher volume relative to price → more liquidity/capital flow
    velocity_raw = volume / close.clip(lower=1e-10)
    df["velocity_raw"] = velocity_raw

    # --- Fast and slow moving averages of velocity ---
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

    # --- Velocity ratio: fast / slow (acceleration signal) ---
    velocity_ratio = velocity_fast / velocity_slow.clip(lower=1e-10)
    df["velocity_ratio"] = velocity_ratio

    # --- Z-score of velocity ratio ---
    df["velocity_zscore"] = rolling_zscore(velocity_ratio, window=config.zscore_window)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    return df
