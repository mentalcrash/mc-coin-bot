"""Entropy-Carry-Momentum 전처리 모듈 (Derivatives).

OHLCV + funding_rate 데이터에서 Shannon entropy, momentum, FR carry feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지). rolling.apply는 entropy 계산에서만 예외적 사용.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

if TYPE_CHECKING:
    from src.strategy.entropy_carry_mom.config import EntropyCarryMomConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "funding_rate"})


def _rolling_shannon_entropy(
    returns: pd.Series,
    window: int,
    bins: int,
) -> pd.Series:
    """Rolling Shannon Entropy 계산.

    각 윈도우에서 수익률의 히스토그램을 구하고,
    scipy.stats.entropy로 Shannon entropy를 계산한다.

    Note:
        rolling.apply()는 scipy.stats.entropy를 벡터화할 수 없기 때문에
        이 함수에서만 예외적으로 사용합니다.

    Args:
        returns: 수익률 시리즈
        window: Rolling 윈도우 크기
        bins: 히스토그램 빈 수

    Returns:
        Rolling Shannon entropy 시리즈
    """

    def _compute_entropy(window_data: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> float:
        """단일 윈도우의 Shannon entropy 계산."""
        hist, _ = np.histogram(window_data, bins=bins)
        # +1e-10 prevents log(0)
        return float(scipy_entropy(hist + 1e-10))

    entropy_series: pd.Series = returns.rolling(
        window=window,
        min_periods=window,
    ).apply(_compute_entropy, raw=True)  # type: ignore[assignment]

    return pd.Series(entropy_series, index=returns.index, name="entropy")


def preprocess(df: pd.DataFrame, config: EntropyCarryMomConfig) -> pd.DataFrame:
    """Entropy-Carry-Momentum feature 계산.

    Calculated columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - entropy: Rolling Shannon entropy
        - entropy_rank: Entropy percentile rank (0~1)
        - mom_direction: Momentum direction (sign of rolling sum)
        - mom_strength: Momentum strength (rolling return magnitude)
        - avg_funding_rate: Rolling mean funding rate
        - fr_zscore: Funding rate z-score
        - carry_direction: -sign(avg_FR) carry direction
        - drawdown: Rolling drawdown (HEDGE_ONLY용)

    Args:
        df: OHLCV + funding_rate DataFrame (DatetimeIndex 필수)
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

    # --- Shannon Entropy ---
    entropy = _rolling_shannon_entropy(
        returns,
        window=config.entropy_window,
        bins=config.entropy_bins,
    )
    df["entropy"] = entropy

    # --- Entropy Percentile Rank (0~1) ---
    df["entropy_rank"] = entropy.rolling(
        window=config.entropy_rank_window,
        min_periods=config.entropy_rank_window,
    ).rank(pct=True)

    # --- Momentum Features ---
    mom_sum: pd.Series = returns.rolling(  # type: ignore[assignment]
        window=config.mom_lookback,
        min_periods=config.mom_lookback,
    ).sum()
    df["mom_direction"] = np.sign(mom_sum)

    mom_ret: pd.Series = returns.rolling(  # type: ignore[assignment]
        window=config.mom_strength_window,
        min_periods=config.mom_strength_window,
    ).sum()
    df["mom_strength"] = mom_ret.abs()

    # --- Funding Rate Features ---
    funding_rate: pd.Series = df["funding_rate"]  # type: ignore[assignment]
    funding_rate = funding_rate.ffill()  # merge_asof 후 NaN 처리

    # Rolling mean FR
    avg_fr: pd.Series = funding_rate.rolling(  # type: ignore[assignment]
        window=config.fr_lookback,
        min_periods=config.fr_lookback,
    ).mean()
    df["avg_funding_rate"] = avg_fr

    # FR z-score
    rolling_mean: pd.Series = avg_fr.rolling(  # type: ignore[assignment]
        window=config.fr_zscore_window,
        min_periods=config.fr_zscore_window,
    ).mean()
    rolling_std: pd.Series = avg_fr.rolling(  # type: ignore[assignment]
        window=config.fr_zscore_window,
        min_periods=config.fr_zscore_window,
    ).std()
    df["fr_zscore"] = (avg_fr - rolling_mean) / rolling_std.clip(lower=1e-10)

    # Carry direction: -sign(avg_FR)
    df["carry_direction"] = pd.Series(-np.sign(avg_fr), index=df.index)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
