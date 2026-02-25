"""Z-Momentum (MACD-V) 전처리 모듈.

OHLCV 데이터에서 MACD-V (ATR-정규화 MACD) feature를 계산한다.
MACD-V = MACD_line / ATR. 변동성 정규화로 고/저 vol 구간 시그널 균질화.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.market.indicators import (
    atr,
    drawdown,
    ema,
    log_returns,
    macd,
    realized_volatility,
    rolling_return,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.z_mom.config import ZMomConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: ZMomConfig) -> pd.DataFrame:
    """Z-Momentum (MACD-V) feature 계산.

    Calculated Columns:
        - returns: Log returns
        - realized_vol: Annualized realized volatility
        - vol_scalar: Vol-target scalar
        - macd_v: MACD-V (= MACD_line / ATR), ATR-정규화 MACD
        - macd_v_signal: MACD-V signal line (EMA of macd_v)
        - macd_v_hist: MACD-V histogram (= macd_v - macd_v_signal)
        - mom_return: Rolling momentum return (confirmation)
        - atr: Average True Range
        - drawdown: Peak drawdown (HEDGE_ONLY용)

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

    # --- MACD ---
    macd_line, _signal_line, _histogram = macd(
        close,
        fast=config.macd_fast,
        slow=config.macd_slow,
        signal=config.macd_signal,
    )

    # --- ATR ---
    atr_series = atr(high, low, close, period=config.atr_period)
    df["atr"] = atr_series

    # --- MACD-V: ATR-정규화 MACD ---
    # MACD-V = MACD_line / ATR (ATR clamp to avoid division by zero)
    atr_clamped = atr_series.clip(lower=1e-10)
    macd_v: pd.Series = macd_line / atr_clamped  # type: ignore[assignment]
    df["macd_v"] = macd_v

    # MACD-V signal line (EMA of MACD-V)
    macd_v_signal = ema(macd_v, span=config.macd_signal)
    df["macd_v_signal"] = macd_v_signal

    # MACD-V histogram
    df["macd_v_hist"] = macd_v - macd_v_signal

    # --- Momentum Return (confirmation) ---
    df["mom_return"] = rolling_return(close, period=config.mom_lookback)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
