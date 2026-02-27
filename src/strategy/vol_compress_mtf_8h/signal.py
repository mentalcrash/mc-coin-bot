"""Volatility Compression Breakout + Multi-TF 시그널 생성 (8H).

변동성 압축 감지 → Donchian 돌파 + 모멘텀 합의 → 진입, 팽창 시 퇴장.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.vol_compress_mtf_8h.config import VolCompressMtf8hConfig


def generate_signals(df: pd.DataFrame, config: VolCompressMtf8hConfig) -> StrategySignals:
    """Volatility Compression Breakout 시그널 생성.

    Signal Logic:
        1. vol_ratio < compression_threshold → compressed 상태
        2. close > prev_dc_upper & mom > 0 → long breakout
        3. close < prev_dc_lower & mom < 0 → short breakout
        4. 진입 = compressed & breakout & momentum 합의
        5. vol_ratio > expansion_threshold → 퇴장 (팽창)
        6. 포지션 유지: 새 시그널 없고 팽창 아니면 ffill

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.vol_compress_mtf_8h.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    vol_ratio = df["vol_ratio"].shift(1)
    mom = df["mom"].shift(1)
    vol_scalar_s = df["vol_scalar"].shift(1)
    dd: pd.Series = df["drawdown"].shift(1)  # type: ignore[assignment]

    # close는 shift하지 않음: 현재 close vs 전봉 Donchian 비교
    close: pd.Series = df["close"]  # type: ignore[assignment]
    prev_dc_upper = df["dc_upper"].shift(1)
    prev_dc_lower = df["dc_lower"].shift(1)

    # --- Compression detection ---
    compressed = vol_ratio < config.compression_threshold

    # --- Breakout confirmation (Donchian) ---
    dc_breakout_up = close > prev_dc_upper
    dc_breakout_dn = close < prev_dc_lower

    # --- Momentum direction (12H approximation via longer lookback on 8H) ---
    mom_up = mom > 0
    mom_dn = mom < 0

    # --- Entry signals: compression + breakout + momentum agreement ---
    long_signal = compressed & dc_breakout_up & mom_up
    short_signal = compressed & dc_breakout_dn & mom_dn

    # --- Expansion exit ---
    expanded = vol_ratio > config.expansion_threshold

    # --- Direction (ShortMode + expansion exit) ---
    direction = _compute_direction_with_exit(
        long_signal=long_signal,
        short_signal=short_signal,
        expanded=expanded,
        dd=dd,
        config=config,
    )

    # --- Strength: compression conviction ---
    # Lower vol_ratio = stronger compression = higher conviction
    compression_strength = (1.0 - vol_ratio.clip(lower=0.0, upper=1.0)).fillna(0)
    strength = direction.astype(float) * compression_strength * vol_scalar_s.fillna(0)

    if config.short_mode == ShortMode.HEDGE_ONLY:
        strength = pd.Series(
            np.where(direction == -1, strength * config.hedge_strength_ratio, strength),
            index=df.index,
        )

    strength = pd.Series(strength, index=df.index).fillna(0.0)

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


def _compute_direction_with_exit(
    long_signal: pd.Series,
    short_signal: pd.Series,
    expanded: pd.Series,
    dd: pd.Series,
    config: VolCompressMtf8hConfig,
) -> pd.Series:
    """ShortMode 3-way 분기 + expansion exit으로 direction 계산.

    포지션 유지(ffill) 후 팽창 구간에서 강제 퇴장한다.
    """
    from src.strategy.vol_compress_mtf_8h.config import ShortMode

    if config.short_mode == ShortMode.DISABLED:
        raw = np.where(long_signal, 1, 0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        hedge_active = dd < config.hedge_threshold
        raw = np.where(
            long_signal,
            1,
            np.where(short_signal & hedge_active, -1, 0),
        )

    else:  # FULL
        raw = np.where(long_signal, 1, np.where(short_signal, -1, 0))

    # Forward-fill to hold position between signals
    dir_series = pd.Series(raw, index=long_signal.index).replace(0, np.nan)
    held = dir_series.ffill().fillna(0).astype(int)

    # Apply expansion exit: force flat when vol expands
    direction = pd.Series(
        np.where(expanded, 0, held),
        index=long_signal.index,
        dtype=int,
    )

    return direction
