"""T-Stat Momentum 전처리 모듈.

OHLCV 데이터에서 multi-lookback t-statistic feature를 계산한다.
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
from src.strategy.t_stat_mom.config import TStatMomConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def _rolling_t_stat(returns: pd.Series, window: int) -> pd.Series:
    """Rolling t-statistic: mean(returns) / (std(returns) / sqrt(N)).

    t-stat는 수익률의 통계적 유의성을 측정한다.
    std/sqrt(N) 정규화로 vol 자동 적응이 내재되어 있다.

    Args:
        returns: log return series
        window: rolling window size

    Returns:
        Rolling t-statistic series
    """
    rolling_mean = returns.rolling(window=window, min_periods=window).mean()
    rolling_std = returns.rolling(window=window, min_periods=window).std()
    # std/sqrt(N) = standard error of the mean
    sqrt_n = np.sqrt(window)
    # 0 나눗셈 방어: std가 0이면 t-stat도 0
    se = rolling_std / sqrt_n
    t_stat: pd.Series = rolling_mean / se.clip(lower=1e-10)  # type: ignore[assignment]
    return t_stat


def preprocess(df: pd.DataFrame, config: TStatMomConfig) -> pd.DataFrame:
    """T-Stat Momentum feature 계산.

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

    # --- Multi-Lookback T-Statistics ---
    df["t_stat_fast"] = _rolling_t_stat(returns, config.fast_lookback)
    df["t_stat_mid"] = _rolling_t_stat(returns, config.mid_lookback)
    df["t_stat_slow"] = _rolling_t_stat(returns, config.slow_lookback)

    # --- Blended T-Stat: equal-weighted average ---
    df["t_stat_blend"] = (df["t_stat_fast"] + df["t_stat_mid"] + df["t_stat_slow"]) / 3.0

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    return df
