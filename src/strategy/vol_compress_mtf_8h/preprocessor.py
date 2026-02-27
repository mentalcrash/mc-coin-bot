"""Volatility Compression Breakout + Multi-TF 전처리 모듈 (8H).

Yang-Zhang 변동성 비율, Donchian 채널, 모멘텀 feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.market.indicators import (
    donchian_channel,
    drawdown,
    log_returns,
    realized_volatility,
    roc,
    volatility_scalar,
    yang_zhang_volatility,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.vol_compress_mtf_8h.config import VolCompressMtf8hConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: VolCompressMtf8hConfig) -> pd.DataFrame:
    """Volatility Compression Breakout feature 계산.

    Calculated Columns:
        - yz_short: 단기 Yang-Zhang 변동성
        - yz_long: 장기 Yang-Zhang 변동성
        - vol_ratio: yz_short / yz_long (압축/팽창 감지)
        - dc_upper, dc_lower: Donchian Channel 상/하단
        - mom: ROC 기반 모멘텀 방향
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

    open_: pd.Series = df["open"]  # type: ignore[assignment]
    close: pd.Series = df["close"]  # type: ignore[assignment]
    high: pd.Series = df["high"]  # type: ignore[assignment]
    low: pd.Series = df["low"]  # type: ignore[assignment]

    # --- Yang-Zhang Volatility (short / long) ---
    yz_short = yang_zhang_volatility(open_, high, low, close, config.yz_short_window)
    yz_long = yang_zhang_volatility(open_, high, low, close, config.yz_long_window)
    df["yz_short"] = yz_short
    df["yz_long"] = yz_long

    # --- Vol Ratio: short / long (safe division) ---
    df["vol_ratio"] = yz_short / yz_long.replace(0, float("nan"))

    # --- Donchian Channel ---
    dc_upper, _dc_mid, dc_lower = donchian_channel(high, low, config.dc_lookback)
    df["dc_upper"] = dc_upper
    df["dc_lower"] = dc_lower

    # --- Momentum (ROC) ---
    df["mom"] = roc(close, config.mom_lookback)

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
