"""Realized Semivariance Momentum 전처리 모듈.

OHLCV 데이터에서 상방/하방 반분산 비율 및 모멘텀 feature를 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

import numpy as np
import pandas as pd

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)
from src.strategy.up_vol_mom.config import UpVolMomConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def _upside_semivariance(returns: pd.Series, window: int) -> pd.Series:
    """Rolling upside semivariance: E[max(r, 0)^2].

    Args:
        returns: 수익률 시리즈
        window: Rolling 윈도우 크기

    Returns:
        Upside semivariance 시리즈
    """
    up_returns = np.maximum(returns, 0.0)
    squared: pd.Series = up_returns**2  # type: ignore[assignment]
    semivar: pd.Series = squared.rolling(  # type: ignore[assignment]
        window=window, min_periods=window
    ).mean()
    return semivar


def _downside_semivariance(returns: pd.Series, window: int) -> pd.Series:
    """Rolling downside semivariance: E[min(r, 0)^2].

    Args:
        returns: 수익률 시리즈
        window: Rolling 윈도우 크기

    Returns:
        Downside semivariance 시리즈
    """
    down_returns = np.minimum(returns, 0.0)
    squared: pd.Series = down_returns**2  # type: ignore[assignment]
    semivar: pd.Series = squared.rolling(  # type: ignore[assignment]
        window=window, min_periods=window
    ).mean()
    return semivar


def preprocess(df: pd.DataFrame, config: UpVolMomConfig) -> pd.DataFrame:
    """Realized Semivariance Momentum feature 계산.

    Calculated Columns:
        - returns: 로그 수익률
        - up_semivar: 상방 반분산
        - down_semivar: 하방 반분산
        - up_ratio: 상방 반분산 / 전체 반분산 (0~1)
        - up_ratio_ma: 상방 비율 이동평균
        - mom_direction: 모멘텀 방향 (+1/0/-1)
        - realized_vol: 실현 변동성
        - vol_scalar: 변동성 스케일러
        - drawdown: 최고점 대비 하락률

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

    # --- Upside / Downside Semivariance ---
    up_sv = _upside_semivariance(returns, window=config.semivar_window)
    down_sv = _downside_semivariance(returns, window=config.semivar_window)
    df["up_semivar"] = up_sv
    df["down_semivar"] = down_sv

    # --- Up Ratio: up_semivar / (up_semivar + down_semivar) ---
    # > 0.5 means upside vol dominant (informed buying)
    # < 0.5 means downside vol dominant (selling pressure)
    total_sv = up_sv + down_sv
    total_sv_safe: pd.Series = total_sv.replace(0, np.nan)  # type: ignore[assignment]
    up_ratio = up_sv / total_sv_safe
    df["up_ratio"] = up_ratio

    # --- Up Ratio Moving Average (smoothing) ---
    up_ratio_ma: pd.Series = up_ratio.rolling(  # type: ignore[assignment]
        window=config.ratio_ma_window, min_periods=config.ratio_ma_window
    ).mean()
    df["up_ratio_ma"] = up_ratio_ma

    # --- Momentum direction ---
    mom_return = close / close.shift(config.mom_lookback) - 1.0
    df["mom_direction"] = np.sign(mom_return)

    # --- Drawdown (HEDGE_ONLY 용) ---
    df["drawdown"] = drawdown(close)

    return df
