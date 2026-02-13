"""Volume-Gated Multi-Scale Momentum 전처리 모듈.

OHLCV 데이터에서 다중 ROC + volume gate feature를 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

import pandas as pd

from src.strategy.tsmom.preprocessor import (
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)
from src.strategy.vmsm.config import VmsmConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: VmsmConfig) -> pd.DataFrame:
    """Volume-Gated Multi-Scale Momentum feature 계산.

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

    # --- Multi-Scale ROC ---
    df["roc_short"] = close / close.shift(config.roc_short) - 1.0
    df["roc_mid"] = close / close.shift(config.roc_mid) - 1.0
    df["roc_long"] = close / close.shift(config.roc_long) - 1.0

    # --- Volume Gate: current volume vs rolling average ---
    vol_avg = volume.rolling(
        window=config.vol_gate_window, min_periods=config.vol_gate_window
    ).mean()
    vol_avg_safe: pd.Series = vol_avg.clip(lower=1.0)  # type: ignore[assignment]
    df["vol_ratio"] = volume / vol_avg_safe

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = calculate_drawdown(close)

    return df
