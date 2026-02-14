"""Vol-Efficiency Momentum 전처리 모듈.

OHLCV 데이터에서 Parkinson vol / CC vol 비율(DVR) 및 모멘텀 feature를 계산.
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
from src.strategy.dvr_mom.config import DvrMomConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})

# Parkinson 상수: 1 / (4 * ln(2))
_PARKINSON_CONST = 1.0 / (4.0 * np.log(2.0))


def preprocess(df: pd.DataFrame, config: DvrMomConfig) -> pd.DataFrame:
    """Vol-Efficiency Momentum feature 계산.

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

    # --- Realized Volatility (Close-to-Close) ---
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

    # --- Parkinson Volatility (High-Low based) ---
    # Parkinson variance per bar: (1/(4*ln2)) * (ln(H/L))^2
    hl_ratio = high / low.clip(lower=1e-10)
    parkinson_var = _PARKINSON_CONST * np.log(hl_ratio) ** 2
    parkinson_vol: pd.Series = np.sqrt(  # type: ignore[assignment]
        parkinson_var.rolling(window=config.dvr_window, min_periods=config.dvr_window).mean()
    )

    # --- Close-to-Close Volatility (same window) ---
    cc_vol = returns.rolling(window=config.dvr_window, min_periods=config.dvr_window).std()

    # --- DVR: CC vol / Parkinson vol ---
    # Low DVR = directional bars (close move captures most of high-low range)
    # High DVR = noisy bars (close move is small vs intrabar range)
    cc_vol_safe = cc_vol.clip(lower=1e-10)
    parkinson_vol_safe = parkinson_vol.clip(lower=1e-10)
    df["dvr"] = cc_vol_safe / parkinson_vol_safe

    # --- Momentum direction ---
    mom_return = close / close.shift(config.mom_lookback) - 1.0
    df["mom_direction"] = np.sign(mom_return)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
