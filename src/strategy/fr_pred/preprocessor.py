"""FR-Pred 전처리 모듈 (Derivatives).

OHLCV + funding_rate 데이터에서 FR 기반 feature 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.market.indicators import (
    drawdown,
    funding_rate_ma,
    funding_zscore,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.fr_pred.config import FRPredConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "funding_rate"})


def preprocess(df: pd.DataFrame, config: FRPredConfig) -> pd.DataFrame:
    """FR-Pred feature 계산 (Derivatives 포함).

    Args:
        df: OHLCV + funding_rate DataFrame.
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

    # Funding Rate features
    funding_rate: pd.Series = df["funding_rate"]  # type: ignore[assignment]
    funding_rate = funding_rate.ffill()  # merge_asof 후 NaN 처리

    # FR MA
    df["fr_ma"] = funding_rate_ma(funding_rate, window=config.fr_ma_window)

    # FR Z-score (mean-reversion signal)
    df["fr_zscore"] = funding_zscore(
        funding_rate,
        ma_window=config.fr_ma_window,
        zscore_window=config.fr_zscore_window,
    )

    # FR Momentum: fast MA - slow MA crossover
    fr_fast_ma = funding_rate.rolling(config.fr_mom_fast).mean()
    fr_slow_ma = funding_rate.rolling(config.fr_mom_slow).mean()
    df["fr_mom_cross"] = fr_fast_ma - fr_slow_ma

    # Drawdown (HEDGE_ONLY용)
    df["drawdown"] = drawdown(close)

    return df
