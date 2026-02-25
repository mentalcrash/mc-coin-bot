"""Squeeze-Adaptive Breakout 전처리 모듈.

OHLCV 데이터에서 squeeze 감지, KAMA 방향, BB position 등 전략 feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.market.indicators import (
    bb_position,
    bollinger_bands,
    drawdown,
    kama,
    keltner_channels,
    log_returns,
    realized_volatility,
    squeeze_detect,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.squeeze_adaptive_breakout.config import SqueezeAdaptiveBreakoutConfig


_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(
    df: pd.DataFrame,
    config: SqueezeAdaptiveBreakoutConfig,
) -> pd.DataFrame:
    """Squeeze-Adaptive Breakout feature 계산.

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

    # --- Bollinger Bands ---
    bb_upper, _bb_mid, bb_lower = bollinger_bands(
        close,
        period=config.bb_period,
        std_dev=config.bb_std,
    )
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower

    # --- Keltner Channels ---
    kc_upper, _kc_mid, kc_lower = keltner_channels(
        high,
        low,
        close,
        ema_period=config.kc_period,
        atr_period=config.kc_atr_period,
        multiplier=config.kc_mult,
    )
    df["kc_upper"] = kc_upper
    df["kc_lower"] = kc_lower

    # --- Squeeze Detection (BB inside KC) ---
    squeeze_on = squeeze_detect(bb_upper, bb_lower, kc_upper, kc_lower)
    df["squeeze_on"] = squeeze_on

    # --- Consecutive Squeeze Count (rolling sum) ---
    squeeze_bool = squeeze_on.astype(int)
    consecutive_squeeze = squeeze_bool.rolling(
        window=config.squeeze_lookback, min_periods=config.squeeze_lookback
    ).sum()
    df["squeeze_duration"] = consecutive_squeeze

    # --- KAMA (Adaptive Direction Indicator) ---
    kama_series = kama(
        close,
        er_lookback=config.kama_er_lookback,
        fast_period=config.kama_fast,
        slow_period=config.kama_slow,
    )
    df["kama"] = kama_series

    # --- BB Position (0~1 normalized conviction) ---
    df["bb_position"] = bb_position(
        close,
        period=config.bb_pos_period,
        std_dev=config.bb_pos_std,
    )

    # --- Drawdown (HEDGE_ONLY 용) ---
    df["drawdown"] = drawdown(close)

    return df
