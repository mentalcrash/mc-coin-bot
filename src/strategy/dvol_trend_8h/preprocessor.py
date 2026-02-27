"""DVOL-Trend 8H 전처리 모듈.

DVOL percentile 기반 size multiplier + 3-scale Donchian Channel + 변동성 스케일러 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.market.indicators import (
    donchian_channel,
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.dvol_trend_8h.config import DvolTrend8hConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: DvolTrend8hConfig) -> pd.DataFrame:
    """DVOL-Trend 8H feature 계산.

    Calculated Columns:
        - dc_upper_{scale}, dc_lower_{scale}: 3-scale Donchian Channel
        - dvol_size_mult: DVOL percentile 기반 size multiplier
        - returns: log return
        - realized_vol: 연환산 실현 변동성
        - vol_scalar: 변동성 스케일러
        - drawdown: HEDGE_ONLY용 drawdown

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수).
            Optional: dvol_close 컬럼 (Deribit 일봉 내재변동성)
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

    # --- DVOL Percentile → Size Multiplier ---
    if "dvol_close" in df.columns:
        dvol: pd.Series = df["dvol_close"].ffill()  # type: ignore[assignment]
        dvol_percentile = dvol.rolling(config.dvol_percentile_window).rank(pct=True)
        df["dvol_size_mult"] = np.where(
            dvol_percentile < config.dvol_low_threshold,
            config.dvol_low_multiplier,
            np.where(
                dvol_percentile > config.dvol_high_threshold,
                config.dvol_high_multiplier,
                1.0,
            ),
        )
    else:
        # DVOL 데이터 없으면 중립 (1.0)
        df["dvol_size_mult"] = 1.0

    # --- 3-Scale Donchian Channels ---
    for scale in (config.dc_scale_short, config.dc_scale_mid, config.dc_scale_long):
        upper, _mid, lower = donchian_channel(high, low, scale)
        df[f"dc_upper_{scale}"] = upper
        df[f"dc_lower_{scale}"] = lower

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

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
