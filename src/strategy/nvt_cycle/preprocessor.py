"""NVT Cycle Signal 전처리 모듈.

OHLCV + on-chain(oc_mktcap_usd, oc_txcnt) 데이터에서 NVT ratio feature 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.nvt_cycle.config import NvtCycleConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)
from src.market.indicators.composite import rolling_zscore

_REQUIRED_COLUMNS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "oc_mktcap_usd",
        "oc_txcnt",
    }
)


def preprocess(df: pd.DataFrame, config: NvtCycleConfig) -> pd.DataFrame:
    """NVT Cycle Signal feature 계산.

    Args:
        df: OHLCV + oc_mktcap_usd + oc_txcnt DataFrame
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

    # --- NVT Ratio ---
    mktcap: pd.Series = df["oc_mktcap_usd"]  # type: ignore[assignment]
    txcnt: pd.Series = df["oc_txcnt"]  # type: ignore[assignment]
    mktcap = mktcap.ffill()
    txcnt = txcnt.ffill()

    # NVT = Market Cap / Transaction Count (smoothed)
    txcnt_safe = txcnt.clip(lower=1.0)
    nvt_raw = mktcap / txcnt_safe

    # Smoothed NVT
    nvt_smooth = nvt_raw.rolling(window=config.nvt_window, min_periods=config.nvt_window).mean()
    df["nvt_ratio"] = nvt_smooth

    # NVT z-score
    df["nvt_zscore"] = rolling_zscore(nvt_smooth, window=config.nvt_zscore_window)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
