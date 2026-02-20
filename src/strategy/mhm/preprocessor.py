"""MHM 전처리 모듈.

다중 horizon 모멘텀 + 역변동성 가중 feature 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    rolling_return,
    volatility_scalar,
)

if TYPE_CHECKING:
    from src.strategy.mhm.config import MHMConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: MHMConfig) -> pd.DataFrame:
    """MHM feature 계산.

    5개 horizon의 rolling return + 역변동성 가중 합산.

    Args:
        df: OHLCV DataFrame.
        config: 전략 설정.

    Returns:
        feature가 추가된 새 DataFrame.

    Raises:
        ValueError: 필수 컬럼 누락 시.
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    df = df.copy()

    close: pd.Series = df["close"]  # type: ignore[assignment]

    # Returns
    returns = log_returns(close)
    df["returns"] = returns

    # Realized Volatility
    realized_vol = realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    df["realized_vol"] = realized_vol

    # Vol Scalar
    df["vol_scalar"] = volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # Multi-Horizon Rolling Returns
    lookbacks = [
        config.lookback_1,
        config.lookback_2,
        config.lookback_3,
        config.lookback_4,
        config.lookback_5,
    ]

    for i, lb in enumerate(lookbacks, 1):
        ret = rolling_return(close, period=lb)
        df[f"mom_{i}"] = ret

        # Per-horizon volatility (for inverse-vol weighting)
        horizon_vol = returns.rolling(lb).std() * np.sqrt(config.annualization_factor)
        df[f"mom_vol_{i}"] = horizon_vol

    # Inverse-vol weighted momentum score
    weighted_sum = pd.Series(0.0, index=df.index)
    weight_sum = pd.Series(0.0, index=df.index)

    for i in range(1, 6):
        mom: pd.Series = df[f"mom_{i}"]  # type: ignore[assignment]
        vol: pd.Series = df[f"mom_vol_{i}"]  # type: ignore[assignment]
        inv_vol = 1.0 / vol.clip(lower=1e-10)
        weighted_sum = weighted_sum + mom * inv_vol
        weight_sum = weight_sum + inv_vol

    df["weighted_mom"] = weighted_sum / weight_sum.clip(lower=1e-10)

    # Agreement: 부호 일치 수 (5개 horizon 중 같은 방향)
    signs = pd.DataFrame(index=df.index)
    for i in range(1, 6):
        signs[f"s_{i}"] = np.sign(df[f"mom_{i}"])

    # Positive/Negative agreement
    df["pos_agreement"] = (signs > 0).sum(axis=1)
    df["neg_agreement"] = (signs < 0).sum(axis=1)
    df["max_agreement"] = pd.concat([df["pos_agreement"], df["neg_agreement"]], axis=1).max(axis=1)

    # Drawdown (HEDGE_ONLY용)
    df["drawdown"] = drawdown(close)

    return df
