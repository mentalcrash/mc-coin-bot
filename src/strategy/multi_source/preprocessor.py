"""Multi-Source 전략 전처리.

SubSignalSpec별 transform을 적용하여 정규화된 서브시그널을 생성합니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.market.indicators import (
    log_returns,
    realized_volatility,
    rolling_zscore,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.multi_source.config import MultiSourceConfig, SubSignalSpec


def preprocess(df: pd.DataFrame, config: MultiSourceConfig) -> pd.DataFrame:
    """데이터 전처리 — 각 서브시그널의 transform 적용 + vol scalar 계산.

    Args:
        df: enriched DataFrame (OHLCV + 추가 컬럼)
        config: MultiSourceConfig 인스턴스

    Returns:
        전처리된 DataFrame (원본 복사)

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    df = df.copy()

    # 필수 컬럼 확인
    missing = {"close"} - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    close: pd.Series = df["close"]  # type: ignore[assignment]

    # Returns & Vol scalar
    returns = log_returns(close)
    df["_returns"] = returns

    realized_vol = realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    df["_realized_vol"] = realized_vol
    df["_vol_scalar"] = volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 각 서브시그널 transform 적용
    for i, spec in enumerate(config.signals):
        col_name = f"_sub_{i}"
        if spec.column not in df.columns:
            df[col_name] = np.nan
            continue

        raw: pd.Series = df[spec.column]  # type: ignore[assignment]
        transformed = _apply_transform(raw, spec)

        if spec.invert:
            transformed = -transformed

        df[col_name] = transformed

    return df


def _apply_transform(series: pd.Series, spec: SubSignalSpec) -> pd.Series:  # type: ignore[type-arg]
    """SubSignalSpec의 transform을 적용.

    Args:
        series: 원본 시리즈
        spec: 서브시그널 사양

    Returns:
        변환된 시리즈
    """
    from src.strategy.multi_source.config import SubSignalTransform

    if spec.transform == SubSignalTransform.ZSCORE:
        return rolling_zscore(series, window=spec.window)

    if spec.transform == SubSignalTransform.PERCENTILE:
        return _rolling_percentile(series, window=spec.window)

    if spec.transform == SubSignalTransform.MA_CROSS:
        return _ma_cross_signal(series, window=spec.window)

    if spec.transform == SubSignalTransform.MOMENTUM:
        return _momentum_signal(series, window=spec.window)

    msg = f"Unknown transform: {spec.transform}"
    raise ValueError(msg)


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:  # type: ignore[type-arg]
    """Rolling percentile rank (0~1)."""

    def _pct_rank(x: pd.Series) -> float:  # type: ignore[type-arg]
        if len(x) < 2:  # noqa: PLR2004
            return np.nan  # type: ignore[return-value]
        return float((x.rank().iloc[-1] - 1) / (len(x) - 1))

    result: pd.Series = series.rolling(window=window, min_periods=window).apply(  # type: ignore[assignment]
        _pct_rank, raw=False
    )
    # 0~1 → -1~1 스케일링
    return result * 2 - 1


def _ma_cross_signal(series: pd.Series, window: int) -> pd.Series:  # type: ignore[type-arg]
    """MA cross signal: (series - SMA(window)) / SMA(window)."""
    sma: pd.Series = series.rolling(window=window, min_periods=window).mean()  # type: ignore[assignment]
    safe_sma = sma.replace(0, np.nan)
    return (series - sma) / safe_sma.abs()


def _momentum_signal(series: pd.Series, window: int) -> pd.Series:  # type: ignore[type-arg]
    """Momentum: pct_change(window) 기반 z-score."""
    mom = series.pct_change(window)
    return rolling_zscore(mom, window=window)
