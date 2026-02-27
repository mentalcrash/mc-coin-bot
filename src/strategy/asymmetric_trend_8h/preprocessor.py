"""Directional-Asymmetric Multi-Scale Momentum 전처리 모듈.

비대칭 lookback의 UP/DOWN ROC 6종과 vol-target feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    roc,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.asymmetric_trend_8h.config import AsymmetricTrend8hConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: AsymmetricTrend8hConfig) -> pd.DataFrame:
    """Directional-Asymmetric Multi-Scale Momentum feature 계산.

    Calculated Columns:
        - up_roc_short, up_roc_mid, up_roc_long: Up-momentum ROC (slow confirmation)
        - dn_roc_short, dn_roc_mid, dn_roc_long: Down-momentum ROC (fast reaction)
        - returns: log return
        - realized_vol: 연환산 실현 변동성
        - vol_scalar: 변동성 스케일러
        - drawdown: HEDGE_ONLY용 drawdown

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

    # --- Up-Momentum ROC (slow confirmation, longer lookbacks) ---
    df["up_roc_short"] = roc(close, config.up_lookback_short)
    df["up_roc_mid"] = roc(close, config.up_lookback_mid)
    df["up_roc_long"] = roc(close, config.up_lookback_long)

    # --- Down-Momentum ROC (fast reaction, shorter lookbacks) ---
    df["dn_roc_short"] = roc(close, config.dn_lookback_short)
    df["dn_roc_mid"] = roc(close, config.dn_lookback_mid)
    df["dn_roc_long"] = roc(close, config.dn_lookback_long)

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

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
