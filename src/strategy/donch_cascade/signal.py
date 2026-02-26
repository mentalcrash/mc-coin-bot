"""Donchian Cascade MTF 시그널 생성.

12H-equivalent Donchian consensus로 방향을 결정하고,
4H EMA confirmation으로 진입 타이밍을 최적화한다.

Cascade Entry Logic:
    1. HTF direction: 3-scale Donchian consensus (donch-multi 동일)
    2. Direction group tracking: 연속 동일 방향 블록 식별
    3. Confirmation: LONG → close > EMA, SHORT → close < EMA
    4. Force entry: max_wait_bars 초과 시 강제 진입
    5. Forward-fill: cummax()로 진입 후 유지

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.donch_cascade.config import DonchCascadeConfig


def generate_signals(df: pd.DataFrame, config: DonchCascadeConfig) -> StrategySignals:
    """Donchian Cascade MTF 시그널 생성.

    Signal Logic:
        1. 각 lookback * htf_multiplier의 Donchian breakout 계산
        2. 3-scale consensus → htf_direction (12H-equivalent)
        3. 4H EMA confirmation으로 진입 타이밍 최적화
        4. max_wait_bars 후 강제 진입

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.donch_cascade.config import ShortMode

    actual_lookbacks = config.actual_lookbacks()

    # --- Shift(1): 전봉 기준 시그널 ---
    vol_scalar = df["vol_scalar"].shift(1)
    dd = df["drawdown"].shift(1)
    confirm_ema = df["confirm_ema"].shift(1)

    # --- Per-scale breakout 시그널 ---
    close: pd.Series = df["close"]  # type: ignore[assignment]
    prev_close = close.shift(1)
    signal_components: list[pd.Series] = []

    for lb in actual_lookbacks:
        prev_upper = df[f"dc_upper_{lb}"].shift(1)
        prev_lower = df[f"dc_lower_{lb}"].shift(1)

        signal_i = pd.Series(
            np.where(
                close > prev_upper,
                1.0,
                np.where(close < prev_lower, -1.0, 0.0),
            ),
            index=df.index,
        )
        signal_components.append(signal_i)

    # --- Consensus: 3-scale 평균 ---
    consensus: pd.Series = pd.concat(signal_components, axis=1).mean(axis=1)  # type: ignore[assignment]

    # --- HTF Direction (12H-equivalent) ---
    dd_series: pd.Series = dd  # type: ignore[assignment]
    htf_direction = _compute_htf_direction(consensus, dd_series, config)

    # --- Cascade Entry: confirmation + force ---
    prev_close_s: pd.Series = prev_close  # type: ignore[assignment]
    confirm_ema_s: pd.Series = confirm_ema  # type: ignore[assignment]
    cascade_direction = _apply_cascade_entry(
        htf_direction=htf_direction,
        prev_close=prev_close_s,
        confirm_ema=confirm_ema_s,
        max_wait_bars=config.max_wait_bars,
    )

    # --- Strength ---
    abs_consensus: pd.Series = consensus.abs()  # type: ignore[assignment]
    strength = cascade_direction.astype(float) * abs_consensus * vol_scalar.fillna(0)

    if config.short_mode == ShortMode.HEDGE_ONLY:
        strength = pd.Series(
            np.where(
                cascade_direction == -1,
                strength * config.hedge_strength_ratio,
                strength,
            ),
            index=df.index,
        )

    strength = strength.fillna(0.0)

    # --- Entries / Exits ---
    prev_dir = cascade_direction.shift(1).fillna(0).astype(int)
    entries = (cascade_direction != 0) & (cascade_direction != prev_dir)
    exits = (cascade_direction == 0) & (prev_dir != 0)

    return StrategySignals(
        entries=entries.astype(bool),
        exits=exits.astype(bool),
        direction=cascade_direction,
        strength=strength,
    )


def _compute_htf_direction(
    consensus: pd.Series,
    drawdown_series: pd.Series,
    config: DonchCascadeConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 HTF direction 계산."""
    from src.strategy.donch_cascade.config import ShortMode

    abs_consensus = consensus.abs()
    above_threshold = abs_consensus >= config.entry_threshold

    long_signal = (consensus > 0) & above_threshold
    short_signal = (consensus < 0) & above_threshold

    if config.short_mode == ShortMode.DISABLED:
        raw = np.where(long_signal, 1, 0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        hedge_active = drawdown_series < config.hedge_threshold
        raw = np.where(
            long_signal,
            1,
            np.where(short_signal & hedge_active, -1, 0),
        )

    else:  # FULL
        raw = np.where(long_signal, 1, np.where(short_signal, -1, 0))

    return pd.Series(raw, index=consensus.index, dtype=int)


def _apply_cascade_entry(
    htf_direction: pd.Series,
    prev_close: pd.Series,
    confirm_ema: pd.Series,
    max_wait_bars: int,
) -> pd.Series:
    """Cascade entry logic: confirmation + force entry + forward-fill.

    벡터화 구현:
        1. Direction group: 연속 동일 방향 블록 식별
        2. bars_in_group: 블록 내 bar 순번 (0-indexed)
        3. Confirmation: LONG → prev_close > EMA, SHORT → prev_close < EMA
        4. Force: bars_in_group >= max_wait_bars
        5. Trigger: (confirmed | forced) & (htf_dir != 0)
        6. Entered: cummax(trigger) within group → 한번 진입 후 유지

    Args:
        htf_direction: HTF 방향 시리즈 (-1, 0, 1)
        prev_close: shift(1) 적용된 종가
        confirm_ema: shift(1) 적용된 confirmation EMA
        max_wait_bars: 확인 대기 최대 bar 수

    Returns:
        cascade direction 시리즈 (-1, 0, 1)
    """
    # Direction group: htf_direction이 변경될 때마다 새 그룹
    dir_changed = htf_direction != htf_direction.shift(1).fillna(0)
    dir_group = dir_changed.cumsum()

    # Bars in group (0-indexed)
    bars_in_group = dir_group.groupby(dir_group).cumcount()

    # Momentum confirmation (shift(1) 이미 적용된 값 사용)
    confirm_long = prev_close > confirm_ema
    confirm_short = prev_close < confirm_ema
    confirmed = ((htf_direction == 1) & confirm_long) | ((htf_direction == -1) & confirm_short)

    # Force entry after max_wait_bars
    forced = bars_in_group >= max_wait_bars

    # Entry trigger: (confirmed OR forced) AND htf_direction is active
    trigger = (confirmed | forced) & (htf_direction != 0)

    # Forward-fill within group: once entered, stay entered
    entered = trigger.groupby(dir_group).cummax().astype(bool)

    # Final direction: htf_direction where entered, 0 otherwise
    cascade_dir = pd.Series(
        np.where(entered, htf_direction, 0),
        index=htf_direction.index,
        dtype=int,
    )

    return cascade_dir
