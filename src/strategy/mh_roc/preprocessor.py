"""Multi-Horizon ROC Ensemble 전처리 모듈.

OHLCV 데이터에서 다중 시간축 ROC feature를 계산.
4개 horizon의 ROC + vote 집계.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.mh_roc.config import MhRocConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: MhRocConfig) -> pd.DataFrame:
    """Multi-Horizon ROC Ensemble feature 계산.

    Calculated columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - roc_short: 단기 ROC
        - roc_medium_short: 중단기 ROC
        - roc_medium_long: 중장기 ROC
        - roc_long: 장기 ROC
        - vote_sum: 4개 ROC 부호 투표 합 (-4 ~ +4)
        - vote_ratio: 정규화 투표 비율 (-1.0 ~ +1.0)
        - drawdown: rolling drawdown (HEDGE_ONLY용)

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

    # --- Multi-Horizon ROC ---
    # ROC = ln(close / close.shift(N)) — 로그 수익률 기반
    lookbacks = [
        ("roc_short", config.roc_short),
        ("roc_medium_short", config.roc_medium_short),
        ("roc_medium_long", config.roc_medium_long),
        ("roc_long", config.roc_long),
    ]

    vote_sum = pd.Series(0.0, index=df.index)
    for col_name, lookback in lookbacks:
        shifted = close.shift(lookback)
        roc: pd.Series = np.log(close / shifted.clip(lower=1e-10))  # type: ignore[assignment]
        df[col_name] = roc
        # 부호 투표: +1 (positive), -1 (negative), 0 (NaN)
        vote_sum = vote_sum + pd.Series(np.sign(roc), index=df.index).fillna(0)

    df["vote_sum"] = vote_sum
    # 정규화 투표 비율: -1.0 ~ +1.0 (4개 horizon)
    num_horizons = 4.0
    df["vote_ratio"] = vote_sum / num_horizons

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
