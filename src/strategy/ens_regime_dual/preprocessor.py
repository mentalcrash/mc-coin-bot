"""Regime-Adaptive Dual-Alpha Ensemble 전처리 모듈.

앙상블 레벨 vol_scalar만 계산. 서브 전략 전처리는 각 전략의 run()에서 수행.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.strategy.ens_regime_dual.config import EnsRegimeDualConfig

from src.market.indicators import log_returns, realized_volatility

_REQUIRED_COLUMNS = frozenset({"close"})
_MIN_VOLATILITY = 0.05


def preprocess(df: pd.DataFrame, config: EnsRegimeDualConfig) -> pd.DataFrame:
    """Ensemble 레벨 전처리 (vol_scalar 계산).

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: 전략 설정

    Returns:
        vol_scalar가 추가된 새 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    df = df.copy()

    # float64 변환
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    close: pd.Series = df["close"]  # type: ignore[assignment]

    # 실현 변동성
    returns = log_returns(close)
    rv = realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    df["realized_vol"] = rv

    # vol_scalar (shift(1): 전봉 변동성 사용)
    clamped_vol = rv.clip(lower=_MIN_VOLATILITY)
    prev_vol = clamped_vol.shift(1)
    df["vol_scalar"] = config.vol_target / prev_vol

    return df
