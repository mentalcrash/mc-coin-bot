"""Macro-Context-Trend 12H preprocessor.

12H EMA cross 추세 + 매크로 risk appetite 컨텍스트.
매크로 데이터(macro_* 컬럼)는 EDA auto-enrich로 merge_asof 제공.
부재 시 graceful degradation (중립 가중치 1.0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.market.indicators import (
    drawdown,
    ema,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.macro_context_trend_12h.config import MacroContextTrendConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: MacroContextTrendConfig) -> pd.DataFrame:
    """모든 feature 계산.

    Args:
        df: OHLCV DataFrame (+ 선택적 macro_* 컬럼).
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

    # --- Returns ---
    returns = log_returns(close)
    df["returns"] = returns

    # --- Realized Volatility + Vol Scalar ---
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

    # --- EMA Cross Trend ---
    df["ema_fast"] = ema(close, span=config.ema_fast)
    df["ema_slow"] = ema(close, span=config.ema_slow)
    df["ema_diff"] = df["ema_fast"] - df["ema_slow"]

    # 추세 방향: fast > slow → +1, fast < slow → -1
    df["trend_direction"] = np.sign(df["ema_diff"])

    # 연속 확인: 동일 방향 연속 bars 카운트
    direction = df["trend_direction"]
    confirm = direction.copy()
    for i in range(1, config.trend_confirm_bars):
        confirm = confirm * (direction == direction.shift(i)).astype(float)
    df["trend_confirmed"] = (confirm != 0).astype(float)

    # --- Macro Context (graceful degradation) ---
    df["macro_context"] = _compute_macro_context(df, config)

    # --- Drawdown for HEDGE_ONLY ---
    df["drawdown"] = drawdown(close)

    return df


def _compute_macro_context(
    df: pd.DataFrame,
    config: MacroContextTrendConfig,
) -> pd.Series:
    """매크로 리스크 선호도 컨텍스트 계산.

    macro_vix, macro_hy_spread 존재 시 risk-off 감지 → 가중치 축소.
    부재 시 중립 가중치 1.0 반환 (graceful degradation).

    Args:
        df: DataFrame (macro_* 컬럼 선택적).
        config: 전략 설정.

    Returns:
        Macro context weight Series (macro_min_weight ~ macro_max_weight).
    """
    import pandas as pd

    has_vix = "macro_vix" in df.columns
    has_hy = "macro_hy_spread" in df.columns

    if not has_vix and not has_hy:
        # Graceful degradation: 매크로 데이터 없으면 중립
        return pd.Series(1.0, index=df.index)

    signals: list[pd.Series] = []
    window = config.macro_window

    if has_vix:
        vix = df["macro_vix"].ffill()
        vix_ma = vix.rolling(window=window, min_periods=1).mean()
        vix_std = vix.rolling(window=window, min_periods=1).std().fillna(1.0)
        vix_zscore = (vix - vix_ma) / vix_std.replace(0, 1.0)
        # VIX 높을수록 risk-off → 가중치 감소
        vix_signal = (-vix_zscore).clip(-2, 2) / 4 + 0.5  # 0~1
        signals.append(vix_signal)

    if has_hy:
        hy = df["macro_hy_spread"].ffill()
        hy_ma = hy.rolling(window=window, min_periods=1).mean()
        hy_std = hy.rolling(window=window, min_periods=1).std().fillna(1.0)
        hy_zscore = (hy - hy_ma) / hy_std.replace(0, 1.0)
        # HY spread 높을수록 risk-off → 가중치 감소
        hy_signal = (-hy_zscore).clip(-2, 2) / 4 + 0.5  # 0~1
        signals.append(hy_signal)

    # 가용 시그널 평균
    avg_signal = pd.concat(signals, axis=1).mean(axis=1)

    # min_weight ~ max_weight 범위로 스케일링
    context_weight = config.macro_min_weight + avg_signal * (
        config.macro_max_weight - config.macro_min_weight
    )

    # macro_risk_weight로 중립(1.0)과 블렌딩
    blended: pd.Series = (  # type: ignore[assignment]
        (1.0 - config.macro_risk_weight) * 1.0 + config.macro_risk_weight * context_weight
    )

    return blended.fillna(1.0)
