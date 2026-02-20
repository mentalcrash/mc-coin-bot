"""F&G Persistence Break 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.

극단 구간(fear/greed zone)에 min_persist일 이상 체류 후 탈출 시 시그널 발생.
Fear zone 탈출 → long, Greed zone 탈출 → short (HEDGE_ONLY).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.fg_persist_break.config import FgPersistBreakConfig


def generate_signals(df: pd.DataFrame, config: FgPersistBreakConfig) -> StrategySignals:
    """F&G Persistence Break 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.fg_persist_break.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    fg = df["oc_fear_greed"].shift(1)
    fear_streak = df["fear_streak"].shift(1)
    greed_streak = df["greed_streak"].shift(1)
    price_mom = df["price_mom"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # 전전봉 streak (break 감지용: 전전봉에는 zone 안, 전봉에는 zone 밖)
    fear_streak_prev = df["fear_streak"].shift(2)
    greed_streak_prev = df["greed_streak"].shift(2)

    # --- Break Detection ---
    # Fear break: 전전봉에 fear zone + min_persist 이상 체류, 전봉에 탈출
    fear_break = (
        (fear_streak_prev >= config.min_persist)
        & (fear_streak == 0)
        & (fg >= config.fear_threshold)
    )

    # Greed break: 전전봉에 greed zone + min_persist 이상 체류, 전봉에 탈출
    greed_break = (
        (greed_streak_prev >= config.min_persist)
        & (greed_streak == 0)
        & (fg <= config.greed_threshold)
    )

    # --- Price Momentum Confirmation ---
    fear_confirmed = fear_break & (price_mom > 0)
    greed_confirmed = greed_break & (price_mom < 0)

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        fear_confirmed=fear_confirmed,
        greed_confirmed=greed_confirmed,
        df=df,
        config=config,
    )

    # --- Strength: streak 길이에 비례 ---
    streak_strength = np.where(
        fear_confirmed,
        np.minimum(fear_streak_prev.fillna(0) / config.max_streak_cap, 1.0),
        np.where(
            greed_confirmed,
            np.minimum(greed_streak_prev.fillna(0) / config.max_streak_cap, 1.0),
            0.0,
        ),
    )
    strength = direction.astype(float) * vol_scalar.fillna(0) * np.maximum(streak_strength, 0.5)

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
    fear_confirmed: pd.Series,
    greed_confirmed: pd.Series,
    df: pd.DataFrame,
    config: FgPersistBreakConfig,
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산.

    Break 시그널은 1-bar 이벤트 → 반대 break까지 hold.
    """
    from src.strategy.fg_persist_break.config import ShortMode

    n = len(df)
    raw = np.zeros(n, dtype=int)

    # Break 이벤트 시 방향 설정 (forward-fill로 hold)
    event = np.where(fear_confirmed, 1, np.where(greed_confirmed, -1, 0))

    # Forward-fill: 마지막 이벤트 방향을 유지
    last_dir = 0
    for i in range(n):
        if event[i] != 0:
            last_dir = int(event[i])
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
