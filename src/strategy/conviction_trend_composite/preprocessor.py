"""Conviction Trend Composite 전처리 모듈.

OHLCV 데이터에서 전략 feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.conviction_trend_composite.config import ConvictionTrendCompositeConfig

from src.market.indicators import (
    drawdown,
    ema,
    log_returns,
    obv,
    realized_volatility,
    rolling_return,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})

# RV ratio threshold: short/long RV < 1.5 means orderly trend environment
_RV_RATIO_THRESHOLD = 1.5


def preprocess(df: pd.DataFrame, config: ConvictionTrendCompositeConfig) -> pd.DataFrame:
    """Conviction Trend Composite feature 계산.

    Calculated columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - price_mom: rolling return (lookback)
        - ema_fast / ema_slow: EMA cross 추세 방향
        - trend_direction: EMA cross 기반 추세 (+1/-1)
        - obv_ema_fast / obv_ema_slow: OBV EMA cross
        - obv_confirms: OBV가 추세를 확인하는지 (bool)
        - rv_ratio: 단기/장기 RV 비율
        - rv_confirms: RV ratio가 추세 환경을 확인하는지 (bool)
        - composite_conviction: OBV + RV 합의 점수 (0~1)
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

    # --- Price Momentum ---
    df["price_mom"] = rolling_return(close, period=config.mom_lookback)

    # --- EMA Cross Trend Direction ---
    fast = ema(close, span=config.mom_fast)
    slow = ema(close, span=config.mom_slow)
    df["ema_fast"] = fast
    df["ema_slow"] = slow
    df["trend_direction"] = np.where(fast > slow, 1, np.where(fast < slow, -1, 0))

    # --- OBV Volume Structure ---
    obv_series = obv(close, volume)
    obv_fast = ema(obv_series, span=config.obv_fast)
    obv_slow = ema(obv_series, span=config.obv_slow)
    df["obv_ema_fast"] = obv_fast
    df["obv_ema_slow"] = obv_slow
    # OBV confirms trend: OBV rising in uptrend, OBV falling in downtrend
    obv_direction = np.where(obv_fast > obv_slow, 1, np.where(obv_fast < obv_slow, -1, 0))
    trend_dir = df["trend_direction"].to_numpy()
    df["obv_confirms"] = (obv_direction == trend_dir) & (trend_dir != 0)

    # --- RV Ratio (short/long) ---
    rv_short = realized_volatility(
        returns,
        window=config.rv_short_window,
        annualization_factor=config.annualization_factor,
    )
    rv_long = realized_volatility(
        returns,
        window=config.rv_long_window,
        annualization_factor=config.annualization_factor,
    )
    rv_ratio: pd.Series = rv_short / rv_long.clip(lower=1e-10)  # type: ignore[assignment]
    df["rv_ratio"] = rv_ratio
    # RV confirms: ratio below threshold means orderly trend environment
    df["rv_confirms"] = rv_ratio < _RV_RATIO_THRESHOLD

    # --- Composite Conviction Score ---
    # 0.5 per source (OBV + RV), total 0~1
    obv_score = np.where(df["obv_confirms"].to_numpy(), 0.5, 0.0)
    rv_score = np.where(df["rv_confirms"].to_numpy(), 0.5, 0.0)
    df["composite_conviction"] = obv_score + rv_score

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
