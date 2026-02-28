"""Vol-Structure-Trend 12H preprocessor.

3종 변동성 추정기(GK/PK/YZ) x 3-scale 앙상블.
각 추정기가 변동성 확대를 탐지 → 합의 기반 추세 진입.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.market.indicators import (
    drawdown,
    garman_klass_volatility,
    log_returns,
    parkinson_volatility,
    realized_volatility,
    roc,
    volatility_scalar,
    yang_zhang_volatility,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.vol_structure_trend_12h.config import VolStructureTrendConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: VolStructureTrendConfig) -> pd.DataFrame:
    """모든 feature 계산.

    Args:
        df: OHLCV DataFrame.
        config: 전략 설정.

    Returns:
        Feature 컬럼이 추가된 DataFrame.
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    df = df.copy()

    close: pd.Series = df["close"]  # type: ignore[assignment]
    high: pd.Series = df["high"]  # type: ignore[assignment]
    low: pd.Series = df["low"]  # type: ignore[assignment]
    opn: pd.Series = df["open"]  # type: ignore[assignment]

    # --- Returns ---
    returns = log_returns(close)
    df["returns"] = returns

    # --- Realized Volatility for vol scalar ---
    realized_vol = realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    df["realized_vol"] = realized_vol
    df["vol_scalar"] = volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # --- Multi-scale GK/PK/YZ 변동성 추정 ---
    scales = (config.scale_short, config.scale_mid, config.scale_long)
    pct_threshold = config.vol_expansion_percentile / 100.0

    # Per-bar 변동성 추정기 (window 없음 — rolling으로 집계)
    gk_raw = garman_klass_volatility(opn, high, low, close)
    pk_raw = parkinson_volatility(high, low)

    for s in scales:
        # GK/PK: per-bar → rolling mean으로 window 집계
        gk = gk_raw.rolling(window=s, min_periods=s).mean()
        pk = pk_raw.rolling(window=s, min_periods=s).mean()
        # YZ: window 파라미터 내장
        yz: pd.Series = yang_zhang_volatility(opn, high, low, close, window=s)  # type: ignore[assignment]

        # 각 추정기의 rolling percentile rank (확대 여부)
        def _pct_rank(x: pd.Series) -> float:
            return float(np.searchsorted(np.sort(x), x.iloc[-1]) / len(x))

        gk_rank = gk.rolling(window=s, min_periods=s).apply(_pct_rank, raw=False)  # type: ignore[union-attr]
        pk_rank = pk.rolling(window=s, min_periods=s).apply(_pct_rank, raw=False)  # type: ignore[union-attr]
        yz_rank = yz.rolling(window=s, min_periods=s).apply(_pct_rank, raw=False)

        # 변동성 확대 동의 비율 (0~1)
        gk_expand = (gk_rank > pct_threshold).astype(float)
        pk_expand = (pk_rank > pct_threshold).astype(float)
        yz_expand = (yz_rank > pct_threshold).astype(float)

        df[f"vol_agreement_{s}"] = (gk_expand + pk_expand + yz_expand) / 3.0

    # --- ROC 모멘텀 방향 ---
    df["roc_direction"] = np.sign(roc(close, period=config.roc_lookback))

    # --- Drawdown for HEDGE_ONLY ---
    df["drawdown"] = drawdown(close)

    return df
