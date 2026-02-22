"""Hash-Ribbon Capitulation 전처리 모듈.

OHLCV 데이터에서 hash ribbon proxy, capitulation/recovery 상태를 계산한다.
실제 hashrate 없이 가격 기반 SMA cross로 capitulation 구간을 프록시한다.
모든 연산은 벡터화 (for 루프 금지).
"""

import numpy as np
import pandas as pd

from src.market.indicators import (
    atr,
    count_consecutive,
    drawdown,
    log_returns,
    realized_volatility,
    sma,
    volatility_scalar,
)
from src.strategy.hash_ribbon_cap.config import HashRibbonCapConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: HashRibbonCapConfig) -> pd.DataFrame:
    """Hash-Ribbon Capitulation feature 계산.

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

    # --- Hash Ribbon Proxy: fast SMA vs slow SMA of close ---
    # Capitulation proxy: fast < slow (매도 압력 구간)
    sma_fast = sma(close, period=config.hash_fast_window)
    sma_slow = sma(close, period=config.hash_slow_window)
    df["sma_fast"] = sma_fast
    df["sma_slow"] = sma_slow

    # capitulation: fast < slow → 1, else 0
    capitulation = (sma_fast < sma_slow).astype(int)
    df["capitulation"] = capitulation

    # recovery zone: fast >= slow (non-capitulation) 연속 bar 수
    not_cap_mask = (capitulation == 0).to_numpy().astype(bool)
    df["recovery_bars"] = pd.Series(
        count_consecutive(not_cap_mask),
        index=df.index,
    )

    # was_in_capitulation: 직전에 capitulation 상태였는지 (recovery 유효성 확인)
    # recovery_bars > 0 이면서 이전 구간에 capitulation이 있었으면 유효
    cap_cummax: pd.Series = capitulation.cummax()  # type: ignore[assignment]
    df["had_capitulation"] = cap_cummax

    # --- Momentum confirmation ---
    mom_return = close / close.shift(config.momentum_lookback) - 1.0
    df["momentum"] = np.sign(mom_return)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    return df
