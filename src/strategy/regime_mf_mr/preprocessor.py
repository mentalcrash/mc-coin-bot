"""Regime-Gated Multi-Factor MR 전처리 모듈.

OHLCV 데이터에서 BB position, Z-score, MR score, RSI, Volume MA를 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

import pandas as pd

from src.market.indicators import (
    bb_position,
    drawdown,
    log_returns,
    mean_reversion_score,
    realized_volatility,
    rolling_zscore,
    rsi,
    volatility_scalar,
)
from src.strategy.regime_mf_mr.config import RegimeMfMrConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: RegimeMfMrConfig) -> pd.DataFrame:
    """Regime-Gated Multi-Factor MR feature 계산.

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

    # --- Multi-Factor MR Features ---
    # 1. BB Position: (close - BB_lower) / (BB_upper - BB_lower), 0~1 범위
    df["bb_pos"] = bb_position(close, period=config.bb_period, std_dev=config.bb_std)

    # 2. Price Z-score
    df["price_zscore"] = rolling_zscore(close, window=config.zscore_window)

    # 3. Mean Reversion Score
    df["mr_score"] = mean_reversion_score(
        close, window=config.mr_score_window, std_mult=config.mr_score_std
    )

    # 4. RSI
    df["rsi"] = rsi(close, period=config.rsi_period)

    # --- Volume Confirmation ---
    df["volume_ma"] = volume.rolling(config.volume_ma_period).mean()

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
