"""Composite Momentum 전처리 모듈.

OHLCV 데이터에서 3축 직교 feature를 계산한다:
1. 가격 모멘텀 z-score (방향)
2. 거래량 z-score (참여도)
3. GK 변동성 z-score (환경)
모든 연산은 벡터화 (for 루프 금지).
"""

import numpy as np
import pandas as pd

from src.market.indicators import (
    atr,
    drawdown,
    garman_klass_volatility,
    log_returns,
    realized_volatility,
    roc,
    rolling_zscore,
    volatility_scalar,
)
from src.strategy.comp_mom.config import CompMomConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: CompMomConfig) -> pd.DataFrame:
    """Composite Momentum feature 계산.

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
    open_: pd.Series = df["open"]  # type: ignore[assignment]
    volume: pd.Series = df["volume"]  # type: ignore[assignment]

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

    # === Axis 1: Price Momentum Z-Score ===
    # ROC captures direction; z-score normalizes across regimes
    price_roc = roc(close, period=config.mom_period)
    df["mom_zscore"] = rolling_zscore(price_roc, window=config.mom_zscore_window)

    # === Axis 2: Volume Z-Score ===
    # Volume participation relative to rolling mean
    df["vol_zscore"] = rolling_zscore(volume, window=config.vol_zscore_window)

    # === Axis 3: GK Volatility Z-Score ===
    # Garman-Klass uses OHLC to efficiently estimate intrabar volatility
    gk_var = garman_klass_volatility(open_, high, low, close)
    gk_rolling: pd.Series = gk_var.rolling(  # type: ignore[assignment]
        window=config.gk_window, min_periods=config.gk_window
    ).mean()
    # Avoid sqrt of negative due to float precision
    gk_vol = pd.Series(np.sqrt(gk_rolling.clip(lower=0.0)), index=df.index)
    df["gk_zscore"] = rolling_zscore(gk_vol, window=config.gk_zscore_window)

    # === Composite Score ===
    # Product of three z-scores: aligned axes amplify, misaligned decay
    # sign(mom_zscore) * abs(mom_zscore) * abs(vol_zscore) * abs(gk_zscore)
    # = mom_zscore * abs(vol_zscore) * abs(gk_zscore)
    # Direction from mom, magnitude from volume & volatility alignment
    mom_z = df["mom_zscore"]
    vol_z = df["vol_zscore"].abs()
    gk_z = df["gk_zscore"].abs()
    df["composite_score"] = mom_z * vol_z * gk_z

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    # --- ATR ---
    df["atr"] = atr(high, low, close, period=config.atr_period)

    return df
