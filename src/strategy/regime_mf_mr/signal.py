"""Regime-Gated Multi-Factor MR 시그널 생성.

Ranging 레짐에서만 MR 시그널을 활성화하는 게이팅 로직.
Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.regime_mf_mr.config import RegimeMfMrConfig


# RegimeService가 주입하는 컬럼 목록
_REGIME_COLUMNS = ("p_trending", "p_ranging", "p_volatile", "regime_label")

# --- Multi-Factor Signal Thresholds ---
_BB_OVERSOLD = 0.2
_BB_OVERBOUGHT = 0.8
_ZSCORE_OVERSOLD = -1.0
_ZSCORE_OVERBOUGHT = 1.0
_MR_SCORE_OVERSOLD = -0.5
_MR_SCORE_OVERBOUGHT = 0.5


def _has_regime_columns(df: pd.DataFrame) -> bool:
    """DataFrame에 regime 컬럼이 있는지 확인."""
    return "p_ranging" in df.columns


def _compute_adaptive_vol_target(
    df: pd.DataFrame,
    config: RegimeMfMrConfig,
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


def generate_signals(df: pd.DataFrame, config: RegimeMfMrConfig) -> StrategySignals:
    """Regime-Gated Multi-Factor MR 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.regime_mf_mr.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    bb_pos = df["bb_pos"].shift(1)
    price_zscore = df["price_zscore"].shift(1)
    mr_score = df["mr_score"].shift(1)
    rsi_val = df["rsi"].shift(1)
    volume_series: pd.Series = df["volume"].shift(1)  # type: ignore[assignment]
    volume_ma = df["volume_ma"].shift(1)

    # --- Regime-Adaptive Vol Target ---
    adaptive_vol_target = _compute_adaptive_vol_target(df, config)
    realized_vol = df["realized_vol"].shift(1)
    clamped_vol = realized_vol.clip(lower=config.min_volatility)
    vol_scalar = adaptive_vol_target / clamped_vol

    # --- Regime Gate: ranging 레짐에서만 활성화 ---
    if _has_regime_columns(df):
        p_ranging = df["p_ranging"].shift(1).fillna(0)
        regime_gate = p_ranging >= config.regime_gate_threshold
    else:
        # regime 컬럼 없으면 항상 활성화 (backward compatible)
        regime_gate = pd.Series(True, index=df.index)

    # --- Volume Confirmation ---
    volume_confirm = volume_series >= (volume_ma * config.volume_threshold)

    # --- Multi-Factor Long Signals (oversold) ---
    long_bb = bb_pos < _BB_OVERSOLD
    long_zscore = price_zscore < _ZSCORE_OVERSOLD
    long_mr = mr_score < _MR_SCORE_OVERSOLD
    long_rsi = rsi_val < config.rsi_oversold

    long_factors = (
        long_bb.astype(int) + long_zscore.astype(int) + long_mr.astype(int) + long_rsi.astype(int)
    )
    long_signal = (long_factors >= config.min_factor_agreement) & volume_confirm & regime_gate

    # --- Multi-Factor Short Signals (overbought) ---
    short_bb = bb_pos > _BB_OVERBOUGHT
    short_zscore = price_zscore > _ZSCORE_OVERBOUGHT
    short_mr = mr_score > _MR_SCORE_OVERBOUGHT
    short_rsi = rsi_val > config.rsi_overbought

    short_factors = (
        short_bb.astype(int)
        + short_zscore.astype(int)
        + short_mr.astype(int)
        + short_rsi.astype(int)
    )
    short_signal = (short_factors >= config.min_factor_agreement) & volume_confirm & regime_gate

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        long_signal=long_signal,
        short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength ---
    strength = direction.astype(float) * vol_scalar.fillna(0)

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
    config: RegimeMfMrConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.regime_mf_mr.config import ShortMode

    if config.short_mode == ShortMode.DISABLED:
        raw = np.where(long_signal, 1, 0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        drawdown_val = df["drawdown"].shift(1)
        hedge_active = drawdown_val < config.hedge_threshold
        raw = np.where(
            long_signal,
            1,
            np.where(short_signal & hedge_active, -1, 0),
        )

    else:  # FULL
        raw = np.where(long_signal, 1, np.where(short_signal, -1, 0))

    return pd.Series(raw, index=df.index, dtype=int)
