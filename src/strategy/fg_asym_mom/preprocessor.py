"""F&G Asymmetric Momentum 전처리 모듈.

OHLCV + on-chain(oc_fear_greed) 데이터에서 비대칭 feature 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.fg_asym_mom.config import FgAsymMomConfig

from src.market.indicators import (
    count_consecutive,
    drawdown,
    log_returns,
    realized_volatility,
    sma,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "oc_fear_greed"})


def preprocess(df: pd.DataFrame, config: FgAsymMomConfig) -> pd.DataFrame:
    """F&G Asymmetric Momentum feature 계산.

    Args:
        df: OHLCV + oc_fear_greed DataFrame (DatetimeIndex 필수)
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

    # --- Fear & Greed ---
    fg: pd.Series = df["oc_fear_greed"]  # type: ignore[assignment]
    fg = fg.ffill()
    df["oc_fear_greed"] = fg

    # --- SMA (가격 확인용) ---
    df["sma_short"] = sma(close, period=config.sma_short)
    df["sma_long"] = sma(close, period=config.sma_long)

    # --- F&G Delta (fear-side 반등 감지용) ---
    df["fg_delta"] = fg - fg.shift(config.fg_delta_window)

    # --- Greed zone streak (persistence break용) ---
    greed_extreme_mask = (fg > config.greed_threshold).to_numpy()
    df["greed_streak"] = count_consecutive(greed_extreme_mask)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
