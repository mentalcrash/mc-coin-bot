"""Carry-Sentiment Gate 전처리 모듈 (Derivatives + On-chain).

OHLCV + funding_rate + oc_fear_greed 데이터에서 전략 feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.carry_sent.config import CarrySentConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    sma,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset(
    {"open", "high", "low", "close", "volume", "funding_rate", "oc_fear_greed"}
)


def preprocess(df: pd.DataFrame, config: CarrySentConfig) -> pd.DataFrame:
    """Carry-Sentiment Gate feature 계산.

    Calculated columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - avg_funding_rate: rolling mean funding rate
        - fr_zscore: funding rate z-score
        - fg_ma: F&G smoothed (SMA)
        - drawdown: rolling drawdown (HEDGE_ONLY용)

    Args:
        df: OHLCV + funding_rate + oc_fear_greed DataFrame (DatetimeIndex 필수)
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

    # --- Funding Rate Features ---
    funding_rate: pd.Series = df["funding_rate"]  # type: ignore[assignment]
    funding_rate = funding_rate.ffill()  # merge_asof 후 NaN 처리

    # Rolling mean FR
    avg_fr: pd.Series = funding_rate.rolling(  # type: ignore[assignment]
        window=config.fr_lookback, min_periods=config.fr_lookback
    ).mean()
    df["avg_funding_rate"] = avg_fr

    # FR z-score
    rolling_mean: pd.Series = avg_fr.rolling(  # type: ignore[assignment]
        window=config.fr_zscore_window, min_periods=config.fr_zscore_window
    ).mean()
    rolling_std: pd.Series = avg_fr.rolling(  # type: ignore[assignment]
        window=config.fr_zscore_window, min_periods=config.fr_zscore_window
    ).std()
    df["fr_zscore"] = (avg_fr - rolling_mean) / rolling_std.clip(lower=1e-10)

    # --- Fear & Greed Feature ---
    fg: pd.Series = df["oc_fear_greed"]  # type: ignore[assignment]
    fg = fg.ffill()  # on-chain 데이터 NaN 처리
    df["oc_fear_greed"] = fg
    df["fg_ma"] = sma(fg, period=config.fg_ma_window)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
