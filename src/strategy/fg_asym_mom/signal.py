"""F&G Asymmetric Momentum 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.

비대칭 접근:
- Fear-side (F&G < fear_threshold): Contrarian buy
  조건: F&G < fear_threshold AND close > SMA_short AND fg_delta > 0
- Greed-side (F&G > greed_hold): Momentum hold
  조건: close > SMA_long → 포지션 유지
- Short: Greed extreme persistence break
  조건: F&G was > greed_threshold for N days, then drops AND close < SMA_short
- Exit: F&G 중립 구간(40-60) 도달 시
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.fg_asym_mom.config import FgAsymMomConfig


def generate_signals(df: pd.DataFrame, config: FgAsymMomConfig) -> StrategySignals:
    """F&G Asymmetric Momentum 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.fg_asym_mom.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    fg = df["oc_fear_greed"].shift(1)
    fg_delta = df["fg_delta"].shift(1)
    close_prev = df["close"].shift(1)
    sma_short = df["sma_short"].shift(1)
    sma_long = df["sma_long"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)
    greed_streak = df["greed_streak"].shift(1)
    greed_streak_prev = df["greed_streak"].shift(2)

    # === Fear-side: Contrarian Buy ===
    fear_buy = (fg < config.fear_threshold) & (close_prev > sma_short) & (fg_delta > 0)

    # === Greed-side: Momentum Hold ===
    greed_hold = (fg > config.greed_hold_threshold) & (close_prev > sma_long)

    # === Short: Greed extreme persistence break ===
    greed_break = (
        (greed_streak_prev >= config.greed_persist_min)
        & (greed_streak == 0)
        & (close_prev < sma_short)
    )

    # === Exit: 중립 구간 도달 (long exit) ===
    neutral_zone = (fg >= config.neutral_low) & (fg <= config.neutral_high)

    # --- Direction 계산 ---
    direction = _compute_direction(
        fear_buy=fear_buy,
        greed_hold=greed_hold,
        greed_break=greed_break,
        neutral_zone=neutral_zone,
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


def _compute_direction(
    fear_buy: pd.Series,
    greed_hold: pd.Series,
    greed_break: pd.Series,
    neutral_zone: pd.Series,
    df: pd.DataFrame,
    config: FgAsymMomConfig,
) -> pd.Series:
    """비대칭 방향 계산 (forward-fill 로직).

    우선순위: fear_buy(+1) > greed_break(-1) > greed_hold(+1 유지) > neutral(0)
    """
    from src.strategy.fg_asym_mom.config import ShortMode

    n = len(df)
    raw = np.zeros(n, dtype=int)

    # numpy 배열로 변환하여 .iloc 오버헤드 제거
    fear_arr = fear_buy.to_numpy()
    greed_brk_arr = greed_break.to_numpy()
    neutral_arr = neutral_zone.to_numpy()
    greed_hld_arr = greed_hold.to_numpy()

    last_dir = 0
    for i in range(n):
        if fear_arr[i]:
            last_dir = 1
        elif greed_brk_arr[i]:
            last_dir = -1
        elif neutral_arr[i] and last_dir != 0:
            last_dir = 0
        elif greed_hld_arr[i] and last_dir >= 0:
            last_dir = 1
        raw[i] = last_dir

    # ShortMode 적용
    if config.short_mode == ShortMode.DISABLED:
        raw = np.where(raw == -1, 0, raw)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        dd = df["drawdown"].shift(1).to_numpy()
        hedge_active = dd < config.hedge_threshold
        raw = np.where(
            (raw == -1) & ~hedge_active,
            0,
            raw,
        )

    return pd.Series(raw, index=df.index, dtype=int)
