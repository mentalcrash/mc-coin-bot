"""Volume-Impulse Momentum 전처리 모듈.

OHLCV 데이터에서 volume spike + directional bar feature를 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

import numpy as np
import pandas as pd

from src.strategy.tsmom.preprocessor import (
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)
from src.strategy.vol_impulse_mom.config import VolImpulseMomConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: VolImpulseMomConfig) -> pd.DataFrame:
    """Volume-Impulse Momentum feature 계산.

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
    open_: pd.Series = df["open"]  # type: ignore[assignment]
    high: pd.Series = df["high"]  # type: ignore[assignment]
    low: pd.Series = df["low"]  # type: ignore[assignment]
    volume: pd.Series = df["volume"]  # type: ignore[assignment]

    # --- Returns ---
    returns = calculate_returns(close)
    df["returns"] = returns

    # --- Realized Volatility ---
    realized_vol = calculate_realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    df["realized_vol"] = realized_vol

    # --- Vol Scalar ---
    df["vol_scalar"] = calculate_volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # --- Volume Spike Detection ---
    vol_avg = volume.rolling(
        window=config.vol_spike_window, min_periods=config.vol_spike_window
    ).mean()
    vol_avg_safe: pd.Series = vol_avg.clip(lower=1.0)  # type: ignore[assignment]
    df["vol_spike_ratio"] = volume / vol_avg_safe

    # --- Body Ratio ---
    candle_range = high - low
    body = (close - open_).abs()
    df["body_ratio"] = pd.Series(
        np.where(candle_range > 0, body / candle_range, 0.0),
        index=df.index,
    )

    # --- Candle Direction ---
    df["candle_dir"] = np.sign(close - open_)

    # --- Impulse: volume spike + directional bar ---
    is_spike = df["vol_spike_ratio"] > config.vol_spike_multiplier
    is_directional = df["body_ratio"] > config.body_ratio_threshold
    impulse_up = is_spike & is_directional & (df["candle_dir"] > 0)
    impulse_down = is_spike & is_directional & (df["candle_dir"] < 0)

    # --- Impulse signal with hold_bars persistence ---
    # Forward fill for hold_bars using rolling max
    impulse_long_raw = impulse_up.astype(float)
    impulse_short_raw = impulse_down.astype(float)
    df["impulse_long"] = impulse_long_raw.rolling(window=config.hold_bars, min_periods=1).max()
    df["impulse_short"] = impulse_short_raw.rolling(window=config.hold_bars, min_periods=1).max()

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = calculate_drawdown(close)

    return df
