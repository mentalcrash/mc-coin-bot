"""Half-Day Momentum-Reversal 전처리 모듈.

OHLCV 데이터에서 반일 수익률, jump score, 모멘텀/리버설 feature 계산.
모든 연산은 벡터화 (for 루프 금지).
"""

import numpy as np
import pandas as pd

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)
from src.strategy.hd_mom_rev.config import HdMomRevConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: HdMomRevConfig) -> pd.DataFrame:
    """Half-Day Momentum-Reversal feature 계산.

    Calculated Columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성
        - vol_scalar: 변동성 스케일러
        - half_return: log(close/open) per 12H bar
        - half_return_smooth: smoothed half_return (MA)
        - jump_score: |half_return| / daily_vol (정규화)
        - is_jump: jump_score >= jump_threshold
        - drawdown: 최고점 대비 하락률

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

    # --- Half Return: log(close/open) per bar ---
    # This captures the intra-bar direction (first half proxy for 12H bars)
    half_return = np.log(close / open_.clip(lower=1e-10))
    df["half_return"] = half_return

    # --- Smoothed half_return ---
    half_return_smooth: pd.Series = half_return.rolling(  # type: ignore[assignment]
        window=config.half_return_ma, min_periods=1
    ).mean()
    df["half_return_smooth"] = half_return_smooth

    # --- Jump Score: normalized magnitude ---
    # |half_return| normalized by per-bar vol (realized_vol / sqrt(annualization_factor))
    per_bar_vol = realized_vol / np.sqrt(config.annualization_factor)
    per_bar_vol_safe = per_bar_vol.clip(lower=1e-10)
    jump_score = half_return.abs() / per_bar_vol_safe
    df["jump_score"] = jump_score

    # --- Is Jump flag ---
    df["is_jump"] = jump_score >= config.jump_threshold

    # --- Drawdown (HEDGE_ONLY 용) ---
    df["drawdown"] = drawdown(close)

    return df
