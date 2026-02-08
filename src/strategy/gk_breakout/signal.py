"""GK Volatility Breakout Signal Generator.

Garman-Klass 변동성 압축 후 Donchian 채널 돌파 시그널을 생성합니다.

Signal Logic:
    1. GK vol ratio < compression_threshold -> 변동성 압축 감지
    2. 압축 상태에서 Donchian Channel 돌파 시 진입
    3. 변동성 확대 (vol ratio >= threshold) 시 포지션 종료

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.gk_breakout.config import GKBreakoutConfig, ShortMode
from src.strategy.types import Direction, StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: GKBreakoutConfig | None = None,
) -> StrategySignals:
    """GK Volatility Breakout 시그널 생성.

    전처리된 DataFrame에서 변동성 압축 + 채널 돌파 시그널을 생성합니다.

    Signal Generation Pipeline:
        1. vol_ratio로 변동성 압축 구간 감지 (shift(1) 적용)
        2. Donchian Channel 돌파 감지 (shift(1) 적용)
        3. 방향 결정: breakout_up -> LONG, breakout_down -> SHORT
        4. 포지션 유지: ffill로 방향 유지, 변동성 확대 시 EXIT
        5. Vol scalar 적용 (변동성 기반 포지션 사이징)
        6. ShortMode 처리
        7. Entry/Exit 시그널 생성

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: vol_ratio, close, dc_upper, dc_lower, vol_scalar
        config: GK Breakout 설정

    Returns:
        StrategySignals NamedTuple

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    if config is None:
        config = GKBreakoutConfig()

    # 입력 검증
    required_cols = {"vol_ratio", "close", "dc_upper", "dc_lower", "vol_scalar"}
    if config.short_mode == ShortMode.HEDGE_ONLY:
        required_cols.add("drawdown")
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # ================================================================
    # 1. Shift(1) 적용 (미래 참조 편향 방지)
    # ================================================================
    # vol_ratio, vol_scalar: 전봉 기준 (현재 봉에서는 아직 알 수 없는 값)
    vol_ratio_prev: pd.Series = df["vol_ratio"].shift(1)  # type: ignore[assignment]
    vol_scalar_prev: pd.Series = df["vol_scalar"].shift(1)  # type: ignore[assignment]

    # Donchian Channel: 전봉까지의 채널 (현재 봉의 가격으로 돌파 확인)
    # close[t] vs dc_upper[t-1]: 현재 종가가 어제까지의 채널 상단을 돌파했는가
    close_series: pd.Series = df["close"]  # type: ignore[assignment]
    upper_prev: pd.Series = df["dc_upper"].shift(1)  # type: ignore[assignment]
    lower_prev: pd.Series = df["dc_lower"].shift(1)  # type: ignore[assignment]

    # ================================================================
    # 2. 변동성 압축 감지
    # ================================================================
    is_compressed = vol_ratio_prev < config.compression_threshold

    # ================================================================
    # 3. Donchian Channel 돌파 감지
    # ================================================================
    breakout_up = is_compressed & (close_series > upper_prev)
    breakout_down = is_compressed & (close_series < lower_prev)

    # ================================================================
    # 4. 방향 결정 + 포지션 유지
    # ================================================================
    # breakout_up -> 1, breakout_down -> -1, 변동성 확대(~is_compressed) -> 0, else -> NaN (유지)
    direction_raw = np.where(
        breakout_up,
        1,
        np.where(
            breakout_down,
            -1,
            np.where(~is_compressed, 0, np.nan),
        ),
    )
    # ffill: 돌파 방향을 유지, 변동성 확대 시 0으로 전환
    direction = pd.Series(direction_raw, index=df.index).ffill().fillna(0).astype(int)

    # ================================================================
    # 5. Strength 계산 (방향 * vol_scalar)
    # ================================================================
    strength = pd.Series(
        direction.astype(float) * vol_scalar_prev.fillna(0),
        index=df.index,
        name="strength",
    )

    # ================================================================
    # 6. ShortMode 처리
    # ================================================================
    if config.short_mode == ShortMode.DISABLED:
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        drawdown_series: pd.Series = df["drawdown"]  # type: ignore[assignment]
        hedge_active = drawdown_series < config.hedge_threshold

        short_mask = direction == Direction.SHORT
        suppress_short = short_mask & ~hedge_active
        direction = direction.where(~suppress_short, Direction.NEUTRAL)
        strength = strength.where(~suppress_short, 0.0)

        active_short = short_mask & hedge_active
        strength = strength.where(
            ~active_short,
            strength * config.hedge_strength_ratio,
        )

        hedge_days = int(hedge_active.sum())
        if hedge_days > 0:
            logger.info(
                "Hedge Mode | Active: %d days (%.1f%%), Threshold: %.1f%%",
                hedge_days,
                hedge_days / len(hedge_active) * 100,
                config.hedge_threshold * 100,
            )

    # ================================================================
    # 7. Entry/Exit 시그널 생성
    # ================================================================
    prev_direction = direction.shift(1).fillna(0).astype(int)

    long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)
    short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)
    entries = pd.Series(long_entry | short_entry, index=df.index, name="entries")

    to_neutral = (direction == Direction.NEUTRAL) & (prev_direction != Direction.NEUTRAL)
    reversal = direction * prev_direction < 0
    exits = pd.Series(to_neutral | reversal, index=df.index, name="exits")

    # NaN 처리
    entries = entries.fillna(False)
    exits = exits.fillna(False)
    direction = pd.Series(direction.fillna(0).astype(int), index=df.index, name="direction")
    strength = pd.Series(strength.fillna(0.0), index=df.index, name="strength")

    # 시그널 통계 로깅
    long_entries = int(long_entry.sum())
    short_entries = int(short_entry.sum())
    total_exits = int(exits.sum())

    if long_entries > 0 or short_entries > 0:
        logger.info(
            "GK Breakout Signals | Long: %d, Short: %d, Exits: %d",
            long_entries,
            short_entries,
            total_exits,
        )

    return StrategySignals(
        entries=entries,
        exits=exits,
        direction=direction,
        strength=strength,
    )
