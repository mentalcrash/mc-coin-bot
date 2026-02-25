"""KAMA Efficiency Trend 전처리 모듈.

OHLCV 데이터에서 KAMA, ER, slope feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.kama_eff_trend.config import KamaEffTrendConfig

from src.market.indicators import (
    atr,
    drawdown,
    efficiency_ratio,
    kama,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: KamaEffTrendConfig) -> pd.DataFrame:
    """KAMA Efficiency Trend feature 계산.

    Calculated Columns:
        - returns: Log returns
        - realized_vol: Annualized realized volatility
        - vol_scalar: Vol-target scalar
        - kama_line: Kaufman Adaptive Moving Average
        - er: Efficiency Ratio (0~1)
        - kama_slope: KAMA rolling slope (방향 판단)
        - kama_dist: (close - kama) / atr (정규화 거리)
        - atr: Average True Range
        - drawdown: Peak drawdown

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

    # --- KAMA ---
    kama_line = kama(
        close,
        er_lookback=config.kama_period,
        fast_period=config.kama_fast,
        slow_period=config.kama_slow,
    )
    df["kama_line"] = kama_line

    # --- Efficiency Ratio ---
    df["er"] = efficiency_ratio(close, period=config.er_period)

    # --- KAMA Slope (rolling change) ---
    kama_slope: pd.Series = kama_line.diff(config.slope_window) / config.slope_window  # type: ignore[assignment]
    df["kama_slope"] = kama_slope

    # --- ATR ---
    atr_series = atr(high, low, close, period=config.atr_period)
    df["atr"] = atr_series

    # --- KAMA-Price Distance (ATR-normalized) ---
    kama_dist: pd.Series = (close - kama_line) / atr_series.clip(lower=1e-10)  # type: ignore[assignment]
    df["kama_dist"] = kama_dist

    # --- Drawdown (HEDGE_ONLY 용) ---
    df["drawdown"] = drawdown(close)

    return df
