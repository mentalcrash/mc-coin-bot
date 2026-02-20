"""Regime-Adaptive Multi-Lookback Momentum 시그널 생성 (레짐 적응형).

RegimeService가 주입하는 regime 컬럼을 활용하여
다중 lookback 모멘텀의 가중치를 레짐 확률에 따라 적응적으로 조절합니다.
regime 컬럼이 없으면 equal weight fallback합니다.

Shift(1) Rule: 모든 feature + regime 컬럼에 shift(1) 적용.

Signal Logic:
    1. 3개 lookback 모멘텀 (fast/mid/slow) 계산
    2. 레짐 확률 기반으로 lookback별 가중치 연속 혼합
       - Trending -> fast 우세 (빠른 추세 반응)
       - Volatile -> slow 우세 (안정적 장기 추세)
       - Ranging -> equal weight (기본값)
    3. blended_momentum > threshold → long, < -threshold → short
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.regime_adaptive_mom.config import RegimeAdaptiveMomConfig

# 기본 가중치 (equal weight, regime 없을 때)
_DEFAULT_FAST_WEIGHT = 1 / 3
_DEFAULT_MID_WEIGHT = 1 / 3
_DEFAULT_SLOW_WEIGHT = 1 / 3


def _has_regime_columns(df: pd.DataFrame) -> bool:
    """DataFrame에 regime 컬럼이 있는지 확인."""
    return "p_trending" in df.columns


def _compute_blended_momentum(
    df: pd.DataFrame,
    config: RegimeAdaptiveMomConfig,
) -> pd.Series:
    """레짐 확률 가중 blended momentum 계산.

    regime 컬럼이 없으면 equal weight로 fallback.
    레짐은 방향이 아닌 시간 스케일 선호도만 결정.
    """
    mom_fast = df["mom_fast"].shift(1)
    mom_mid = df["mom_mid"].shift(1)
    mom_slow = df["mom_slow"].shift(1)

    if not _has_regime_columns(df):
        # Equal weight fallback
        blended: pd.Series = (  # type: ignore[assignment]
            _DEFAULT_FAST_WEIGHT * mom_fast
            + _DEFAULT_MID_WEIGHT * mom_mid
            + _DEFAULT_SLOW_WEIGHT * mom_slow
        )
        return blended

    # Regime probability weighted
    p_trending = df["p_trending"].shift(1).fillna(1 / 3)
    p_ranging = df["p_ranging"].shift(1).fillna(1 / 3)
    p_volatile = df["p_volatile"].shift(1).fillna(1 / 3)

    # Trending weights
    w_fast_t = config.trending_fast_weight
    w_mid_t = config.trending_mid_weight
    w_slow_t = config.trending_slow_weight

    # Volatile weights
    w_fast_v = config.volatile_fast_weight
    w_mid_v = config.volatile_mid_weight
    w_slow_v = config.volatile_slow_weight

    # Ranging: equal weight
    w_fast_r = _DEFAULT_FAST_WEIGHT
    w_mid_r = _DEFAULT_MID_WEIGHT
    w_slow_r = _DEFAULT_SLOW_WEIGHT

    # Probability-weighted blending of weights
    w_fast = p_trending * w_fast_t + p_ranging * w_fast_r + p_volatile * w_fast_v
    w_mid = p_trending * w_mid_t + p_ranging * w_mid_r + p_volatile * w_mid_v
    w_slow = p_trending * w_slow_t + p_ranging * w_slow_r + p_volatile * w_slow_v

    blended = w_fast * mom_fast + w_mid * mom_mid + w_slow * mom_slow  # type: ignore[assignment]
    return blended


def _compute_adaptive_vol_target(
    df: pd.DataFrame,
    config: RegimeAdaptiveMomConfig,
) -> pd.Series:
    """레짐 확률 가중 adaptive vol_target 계산."""
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


def generate_signals(df: pd.DataFrame, config: RegimeAdaptiveMomConfig) -> StrategySignals:
    """Regime-Adaptive Multi-Lookback Momentum 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.regime_adaptive_mom.config import ShortMode

    # --- Blended Momentum ---
    blended_mom = _compute_blended_momentum(df, config)

    # --- Regime-Adaptive Vol Target ---
    adaptive_vol_target = _compute_adaptive_vol_target(df, config)

    # --- Vol Scalar (adaptive) ---
    realized_vol = df["realized_vol"].shift(1)
    clamped_vol = realized_vol.clip(lower=config.min_volatility)
    vol_scalar = adaptive_vol_target / clamped_vol

    # --- Signal Logic ---
    long_signal = blended_mom > config.signal_threshold
    short_signal = blended_mom < -config.signal_threshold

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength ---
    # Momentum magnitude as conviction (normalized, clamped 0~1)
    mom_magnitude = blended_mom.abs().clip(upper=0.5).fillna(0) / 0.5
    mom_magnitude = mom_magnitude.clip(lower=0.3)  # min conviction when active

    strength = direction.astype(float) * vol_scalar.fillna(0) * mom_magnitude

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
    config: RegimeAdaptiveMomConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.regime_adaptive_mom.config import ShortMode

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
