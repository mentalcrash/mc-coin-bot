"""Cascade Momentum 전처리 모듈.

OHLCV 데이터에서 연속 동방향 streak 기반 cascade score를 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

import numpy as np
import pandas as pd

from src.strategy.cascade_mom.config import CascadeMomConfig
from src.strategy.tsmom.preprocessor import (
    calculate_atr,
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def _compute_streak(direction: pd.Series) -> pd.Series:
    """연속 동방향 bar streak 계산 (벡터화).

    방향이 바뀌면 0으로 리셋. 양수 = 연속 상승, 음수 = 연속 하락.

    Args:
        direction: +1/-1/0 방향 시리즈

    Returns:
        signed streak 시리즈
    """
    # Direction change detection
    changed = direction != direction.shift(1)
    # Group ID for consecutive same-direction bars
    group_id = changed.cumsum()
    # Cumulative count within each group
    cumcount = group_id.groupby(group_id).cumcount() + 1
    # Sign by direction
    return cumcount * direction


def preprocess(df: pd.DataFrame, config: CascadeMomConfig) -> pd.DataFrame:
    """Cascade Momentum feature 계산.

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
    open_: pd.Series = df["open"]  # type: ignore[assignment]
    high: pd.Series = df["high"]  # type: ignore[assignment]
    low: pd.Series = df["low"]  # type: ignore[assignment]

    # --- Returns ---
    returns = calculate_returns(close)
    df["returns"] = returns

    # --- Realized Volatility ---
    realized_vol = calculate_realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    df["realized_vol"] = realized_vol

    # --- Vol Scalar ---
    df["vol_scalar"] = calculate_volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # --- ATR ---
    atr = calculate_atr(high, low, close, period=config.atr_period)

    # --- Bar direction ---
    bar_dir = pd.Series(np.sign(close - open_), index=df.index)

    # --- Streak ---
    streak = _compute_streak(bar_dir)

    # --- Body / ATR ratio ---
    body = (close - open_).abs()
    atr_safe = atr.clip(lower=1e-10)
    body_atr = body / atr_safe

    # --- Cascade Score: streak * rolling_mean(body/ATR) over cascade_window ---
    rolling_body_atr = body_atr.rolling(
        window=config.cascade_window, min_periods=config.cascade_window
    ).mean()
    df["cascade_score"] = streak * rolling_body_atr

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = calculate_drawdown(close)

    return df
