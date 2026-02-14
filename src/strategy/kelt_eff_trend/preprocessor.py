"""Keltner Efficiency Trend 전처리 모듈.

OHLCV 데이터에서 Keltner Channel과 Efficiency Ratio feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.kelt_eff_trend.config import KeltEffTrendConfig

from src.market.indicators import (
    drawdown,
    efficiency_ratio,
    keltner_channels,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: KeltEffTrendConfig) -> pd.DataFrame:
    """Keltner Efficiency Trend feature 계산.

    Calculated columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - kc_upper: Keltner Channel 상단
        - kc_middle: Keltner Channel 중앙 (EMA)
        - kc_lower: Keltner Channel 하단
        - efficiency_ratio: Efficiency Ratio (0~1)
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

    # --- Keltner Channels ---
    kc_upper, kc_middle, kc_lower = keltner_channels(
        high,
        low,
        close,
        ema_period=config.kc_ema_period,
        atr_period=config.kc_atr_period,
        multiplier=config.kc_multiplier,
    )
    df["kc_upper"] = kc_upper
    df["kc_middle"] = kc_middle
    df["kc_lower"] = kc_lower

    # --- Efficiency Ratio ---
    df["efficiency_ratio"] = efficiency_ratio(close, period=config.er_period)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
