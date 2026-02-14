"""FR Conditional Momentum 전처리 모듈 (Derivatives).

OHLCV + funding_rate 데이터에서 모멘텀 + FR conviction feature 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

import numpy as np
import pandas as pd

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    rolling_zscore,
    volatility_scalar,
)
from src.strategy.fr_cond_mom.config import FrCondMomConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "funding_rate"})


def preprocess(df: pd.DataFrame, config: FrCondMomConfig) -> pd.DataFrame:
    """FR Conditional Momentum feature 계산.

    Calculated Columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성
        - vol_scalar: 변동성 스케일러
        - mom_signal: smoothed 모멘텀 시그널 (sign of return MA)
        - fr_ma: funding rate 이동평균
        - fr_zscore: funding rate z-score
        - fr_conviction: FR 기반 conviction 조절 계수 (0~1)
        - drawdown: 최고점 대비 하락률

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

    # --- Momentum Signal ---
    # Rolling return over lookback, then smooth with MA
    mom_return = close / close.shift(config.mom_lookback) - 1.0
    mom_ma: pd.Series = mom_return.rolling(  # type: ignore[assignment]
        window=config.mom_ma_window, min_periods=config.mom_ma_window
    ).mean()
    df["mom_signal"] = np.sign(mom_ma)

    # --- Funding Rate Features ---
    funding_rate: pd.Series = df["funding_rate"]  # type: ignore[assignment]
    funding_rate = funding_rate.ffill()  # merge_asof 후 NaN 처리

    # FR moving average
    fr_ma: pd.Series = funding_rate.rolling(  # type: ignore[assignment]
        window=config.fr_ma_window, min_periods=config.fr_ma_window
    ).mean()
    df["fr_ma"] = fr_ma

    # FR z-score
    fr_zs = rolling_zscore(fr_ma, window=config.fr_zscore_window)
    df["fr_zscore"] = fr_zs

    # --- FR Conviction: dampening at extremes ---
    # |z| <= neutral_zone -> conviction = 1.0 (full conviction)
    # |z| >= extreme_threshold -> conviction = fr_dampening (minimum)
    # between -> linear interpolation
    abs_zscore = fr_zs.abs()
    zone_width = config.fr_extreme_threshold - config.fr_neutral_zone

    # Linear interpolation between neutral and extreme
    interp_ratio = ((abs_zscore - config.fr_neutral_zone) / zone_width).clip(lower=0.0, upper=1.0)
    conviction = 1.0 - interp_ratio * (1.0 - config.fr_dampening)
    df["fr_conviction"] = conviction

    # --- Drawdown (HEDGE_ONLY 용) ---
    df["drawdown"] = drawdown(close)

    return df
