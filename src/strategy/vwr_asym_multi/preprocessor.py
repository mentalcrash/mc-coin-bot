"""VWR Asymmetric Multi-Scale 전처리 모듈.

3-scale Volume-Weighted Returns + z-score + 변동성 스케일러를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
    volume_weighted_returns,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.vwr_asym_multi.config import VwrAsymMultiConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def _rolling_zscore(
    series: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling z-score 계산.

    Args:
        series: 입력 시리즈.
        window: rolling window 크기.

    Returns:
        z-score 시리즈.
    """
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    # 0 나눗셈 방지
    rolling_std_safe = rolling_std.clip(lower=1e-10)
    result: pd.Series = (series - rolling_mean) / rolling_std_safe  # type: ignore[assignment]
    return result


def preprocess(df: pd.DataFrame, config: VwrAsymMultiConfig) -> pd.DataFrame:
    """VWR Asymmetric Multi-Scale feature 계산.

    Calculated Columns:
        - returns: log return
        - realized_vol: 연환산 실현 변동성
        - vol_scalar: 변동성 스케일러
        - vwr_{lb}: 3-scale Volume-Weighted Returns
        - vwr_zscore_{lb}: 3-scale VWR z-score
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

    close: pd.Series = df["close"]  # type: ignore[assignment]
    volume: pd.Series = df["volume"]  # type: ignore[assignment]

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

    # --- 3-Scale Volume-Weighted Returns + Z-Scores ---
    lookbacks = (config.lookback_short, config.lookback_mid, config.lookback_long)
    for lb in lookbacks:
        vwr = volume_weighted_returns(returns, volume, window=lb)
        df[f"vwr_{lb}"] = vwr
        df[f"vwr_zscore_{lb}"] = _rolling_zscore(vwr, window=config.zscore_window)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
