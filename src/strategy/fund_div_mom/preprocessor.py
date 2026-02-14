"""Funding Divergence Momentum 전처리 모듈 (Derivatives).

OHLCV + funding_rate 데이터에서 가격 모멘텀과 FR 추세의 divergence feature를 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.fund_div_mom.config import FundDivMomConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "funding_rate"})


def preprocess(df: pd.DataFrame, config: FundDivMomConfig) -> pd.DataFrame:
    """Funding Divergence Momentum feature 계산.

    Calculated columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - price_mom: 가격 ROC 모멘텀
        - avg_funding_rate: rolling mean funding rate
        - fr_zscore: funding rate z-score
        - fr_direction: FR 추세 방향 (부호)
        - divergence_score: 가격 mom 부호 vs FR z-score 부호 divergence
        - drawdown: rolling drawdown (HEDGE_ONLY용)

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

    # --- Price Momentum (ROC) ---
    # 단순 price rate-of-change over mom_lookback
    shifted_close = close.shift(config.mom_lookback)
    price_mom: pd.Series = np.log(close / shifted_close.clip(lower=1e-10))  # type: ignore[assignment]
    df["price_mom"] = price_mom

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

    # FR 추세 방향 (z-score의 부호)
    fr_zscore: pd.Series = df["fr_zscore"]  # type: ignore[assignment]
    df["fr_direction"] = np.sign(fr_zscore)

    # --- Divergence Score ---
    # 가격 모멘텀 방향과 FR 추세 방향의 divergence
    # 가격 상승(+) + FR 하락(-) = 유기적 수요 → positive divergence (long bias)
    # 가격 하락(-) + FR 상승(+) = 투기적 숏 → negative divergence (short bias)
    # 동일 방향이면 divergence 약함
    mom_sign = pd.Series(np.sign(price_mom), index=df.index)
    fr_sign = pd.Series(np.sign(fr_zscore), index=df.index)

    # divergence_score = price_mom_sign * (1 - fr_sign * mom_sign)
    # 같은 방향(+*+ or -*-): 1-1=0 → no divergence
    # 다른 방향(+*- or -*+): 1-(-1)=2 → strong divergence
    # mom_sign으로 부호 결정: 가격 상승 divergence → positive
    agreement = fr_sign * mom_sign
    divergence_raw = mom_sign * (1 - agreement)
    # |price_mom| 크기로 conviction 스케일링
    mom_abs = price_mom.abs()
    mom_norm_window = config.fr_zscore_window
    rolling_mom_std: pd.Series = mom_abs.rolling(  # type: ignore[assignment]
        window=mom_norm_window, min_periods=mom_norm_window
    ).std()
    mom_normalized = mom_abs / rolling_mom_std.clip(lower=1e-10)

    df["divergence_score"] = divergence_raw * mom_normalized.clip(upper=5.0)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
