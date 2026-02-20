"""Conviction Trend Composite 시그널 생성 (레짐 적응형).

RegimeService가 주입하는 regime 컬럼을 활용하여
vol_target을 레짐 확률에 따라 적응적으로 조절합니다.
regime 컬럼이 없으면 기본 config.vol_target으로 fallback합니다.

Shift(1) Rule: 모든 feature + regime 컬럼에 shift(1) 적용.

Signal Logic:
    1. 가격 모멘텀(EMA cross + rolling return)이 방향 결정
    2. OBV 거래량 구조 + RV ratio가 composite conviction score
    3. conviction_threshold 이상일 때만 진입
    4. 레짐 확률 가중 vol_target으로 사이징 조절
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.conviction_trend_composite.config import ConvictionTrendCompositeConfig


def _has_regime_columns(df: pd.DataFrame) -> bool:
    """DataFrame에 regime 컬럼이 있는지 확인."""
    return "p_trending" in df.columns


def _compute_adaptive_vol_target(
    df: pd.DataFrame,
    config: ConvictionTrendCompositeConfig,
) -> pd.Series:
    """레짐 확률 가중 adaptive vol_target 계산.

    regime 컬럼이 없으면 config.vol_target 상수 반환.
    """
    if not _has_regime_columns(df):
        return pd.Series(config.vol_target, index=df.index)

    p_trending = df["p_trending"].shift(1).fillna(1 / 3)
    p_ranging = df["p_ranging"].shift(1).fillna(1 / 3)
    p_volatile = df["p_volatile"].shift(1).fillna(1 / 3)

    adaptive: pd.Series = (  # type: ignore[assignment]
        p_trending * config.trending_vol_target
        + p_ranging * config.ranging_vol_target
        + p_volatile * config.volatile_vol_target
    )
    return adaptive


def generate_signals(df: pd.DataFrame, config: ConvictionTrendCompositeConfig) -> StrategySignals:
    """Conviction Trend Composite 시그널 생성 (레짐 적응형).

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.conviction_trend_composite.config import ShortMode

    # --- Regime-Adaptive Vol Target ---
    adaptive_vol_target = _compute_adaptive_vol_target(df, config)

    # --- Vol Scalar (adaptive) ---
    realized_vol = df["realized_vol"].shift(1)
    clamped_vol = realized_vol.clip(lower=config.min_volatility)
    vol_scalar = adaptive_vol_target / clamped_vol

    # --- Shift(1): 전봉 기준 시그널 ---
    trend_dir = df["trend_direction"].shift(1)
    price_mom = df["price_mom"].shift(1)
    conviction = df["composite_conviction"].shift(1)

    # --- Signal Logic ---
    # Direction from EMA cross + momentum confirmation
    mom_long = (trend_dir > 0) & (price_mom > 0)
    mom_short = (trend_dir < 0) & (price_mom < 0)

    # Conviction gate: only enter when composite conviction >= threshold
    conviction_gate = conviction >= config.conviction_threshold

    long_signal = mom_long & conviction_gate
    short_signal = mom_short & conviction_gate

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength ---
    # conviction modulates strength (0~1 range)
    conviction_mod = conviction.fillna(0).clip(lower=0.0, upper=1.0)
    strength = direction.astype(float) * vol_scalar.fillna(0) * conviction_mod

    if config.short_mode == ShortMode.HEDGE_ONLY:
        strength = pd.Series(
            np.where(direction == -1, strength * config.hedge_strength_ratio, strength),
            index=df.index,
        )

    strength = strength.fillna(0.0)

    # --- Entries / Exits ---
    prev_dir = direction.shift(1).fillna(0).astype(int)
    entries = (direction != 0) & (direction != prev_dir)
    exits = (direction == 0) & (prev_dir != 0)

    return StrategySignals(
        entries=entries.astype(bool),
        exits=exits.astype(bool),
        direction=direction,
        strength=strength,
    )


def _compute_direction(
    long_signal: pd.Series,
    short_signal: pd.Series,
    df: pd.DataFrame,
    config: ConvictionTrendCompositeConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.conviction_trend_composite.config import ShortMode

    if config.short_mode == ShortMode.DISABLED:
        raw = np.where(long_signal, 1, 0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        dd = df["drawdown"].shift(1)
        hedge_active = dd < config.hedge_threshold
        raw = np.where(
            long_signal,
            1,
            np.where(short_signal & hedge_active, -1, 0),
        )

    else:  # FULL
        raw = np.where(long_signal, 1, np.where(short_signal, -1, 0))

    return pd.Series(raw, index=df.index, dtype=int)
