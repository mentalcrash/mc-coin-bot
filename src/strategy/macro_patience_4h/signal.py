"""Macro-Gated Patient Trend (4H) — signal generation.

Core logic:
1. Multi-scale Donchian consensus → channel direction
2. Macro composite z-score → direction gate
3. Final signal = channel direction ONLY when macro confirms
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.macro_patience_4h.config import MacroPatience4hConfig


def generate_signals(df: pd.DataFrame, config: MacroPatience4hConfig) -> StrategySignals:
    """Generate macro-gated trend signals."""
    from src.strategy.macro_patience_4h.config import ShortMode

    scales = (config.dc_scale_short, config.dc_scale_mid, config.dc_scale_long)
    close: pd.Series = df["close"]  # type: ignore[assignment]

    # --- Shift(1): use previous bar's indicators ---
    vol_scalar = df["vol_scalar"].shift(1)
    macro_dir: pd.Series = df["macro_direction"].shift(1).fillna(0).astype(int)  # type: ignore[assignment]
    dd: pd.Series = df["drawdown"].shift(1)  # type: ignore[assignment]

    # --- Multi-scale Donchian consensus ---
    signal_components: list[pd.Series] = []
    for scale in scales:
        prev_upper = df[f"dc_upper_{scale}"].shift(1)
        prev_lower = df[f"dc_lower_{scale}"].shift(1)
        sig = pd.Series(
            np.where(
                close > prev_upper,
                1.0,
                np.where(close < prev_lower, -1.0, 0.0),
            ),
            index=df.index,
        )
        signal_components.append(sig)

    consensus: pd.Series = pd.concat(signal_components, axis=1).mean(axis=1)  # type: ignore[assignment]

    # --- Macro gate: only allow trades in macro-confirmed direction ---
    gated_consensus = _apply_macro_gate(consensus, macro_dir, config)

    # --- Direction (ShortMode 3-way) ---
    direction = _compute_direction(gated_consensus, dd, config)

    # --- Strength ---
    abs_consensus: pd.Series = gated_consensus.abs()
    strength = direction.astype(float) * abs_consensus * vol_scalar.fillna(0)
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


def _apply_macro_gate(
    consensus: pd.Series, macro_dir: pd.Series, config: MacroPatience4hConfig
) -> pd.Series:
    """Apply macro direction gate to channel consensus.

    - macro_dir == +1 (risk-on): allow long signals only
    - macro_dir == -1 (risk-off): allow short signals only
    - macro_dir == 0 (neutral): allow all signals (no gate)
    """
    return pd.Series(
        np.where(
            macro_dir == 0,
            consensus,  # neutral: pass through
            np.where(
                np.sign(consensus) == macro_dir,
                consensus,  # aligned: pass through
                0.0,  # misaligned: suppress
            ),
        ),
        index=consensus.index,
    )


def _compute_direction(
    consensus: pd.Series, drawdown_series: pd.Series, config: MacroPatience4hConfig
) -> pd.Series:
    """Compute position direction based on gated consensus and ShortMode."""
    from src.strategy.macro_patience_4h.config import ShortMode

    abs_consensus = consensus.abs()
    above_threshold = abs_consensus >= config.entry_threshold

    long_signal = (consensus > 0) & above_threshold
    short_signal = (consensus < 0) & above_threshold

    if config.short_mode == ShortMode.DISABLED:
        raw = np.where(long_signal, 1, 0)
    elif config.short_mode == ShortMode.HEDGE_ONLY:
        hedge_active = drawdown_series < config.hedge_threshold
        raw = np.where(long_signal, 1, np.where(short_signal & hedge_active, -1, 0))
    else:  # FULL
        raw = np.where(long_signal, 1, np.where(short_signal, -1, 0))

    return pd.Series(raw, index=consensus.index, dtype=int)
