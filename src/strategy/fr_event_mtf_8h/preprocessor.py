"""Funding Rate Event Trigger + 12H Momentum Context 전처리 모듈 (8H TF).

FR z-score, EMA 추세, 변동성 feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.market.indicators import (
    drawdown,
    ema,
    funding_zscore,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.fr_event_mtf_8h.config import FrEventMtf8hConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: FrEventMtf8hConfig) -> pd.DataFrame:
    """FR Event MTF 8H feature 계산.

    Calculated Columns:
        - fr_zscore: 펀딩비 z-score (극단 포지셔닝 감지)
        - ema_fast: 빠른 EMA (추세 방향)
        - ema_slow: 느린 EMA (추세 방향)
        - returns: log return
        - realized_vol: 연환산 실현 변동성
        - vol_scalar: 변동성 스케일러
        - drawdown: HEDGE_ONLY용 drawdown

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수).
            deriv_funding_rate 컬럼이 있으면 사용, 없으면 0.0 fallback.
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

    # --- Funding Rate z-score (graceful degradation: 없으면 0.0) ---
    if "deriv_funding_rate" in df.columns:
        fr_series: pd.Series = df["deriv_funding_rate"].fillna(0.0)  # type: ignore[assignment]
    else:
        import pandas as _pd

        fr_series = _pd.Series(0.0, index=df.index)

    df["fr_zscore"] = funding_zscore(fr_series, config.fr_ma_window, config.fr_zscore_window)

    # --- EMA Trend ---
    df["ema_fast"] = ema(close, config.ema_fast_period)
    df["ema_slow"] = ema(close, config.ema_slow_period)

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
