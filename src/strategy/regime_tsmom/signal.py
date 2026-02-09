"""Regime-Adaptive TSMOM Signal Generator.

레짐 확률에 따라 vol_target을 적응적으로 조절합니다.
trending → 공격적, ranging → 보수적, volatile → 초보수.

Signal Formula:
    1. momentum_direction = sign(vw_momentum)
    2. adaptive_vol_target = p_trending * tv + p_ranging * rv + p_volatile * vv
    3. vol_scalar = adaptive_vol_target / realized_vol
    4. strength = direction * vol_scalar  (shifted by 1)

레짐 적응은 vol_target 단일 채널로만 수행합니다.
leverage_scale 이중 곱셈은 과도한 포지션 축소를 유발하므로 제거.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.tsmom.config import ShortMode
from src.strategy.types import Direction, StrategySignals

if TYPE_CHECKING:
    from src.strategy.regime_tsmom.config import RegimeTSMOMConfig


def generate_signals(
    df: pd.DataFrame,
    config: RegimeTSMOMConfig,
) -> StrategySignals:
    """Regime-Adaptive TSMOM 시그널 생성.

    전처리된 DataFrame에서 레짐 확률을 가중하여 시그널을 계산합니다.
    vol_target만 레짐에 따라 적응 — leverage_scale 이중 곱셈 없음.

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: vw_momentum, realized_vol,
                       p_trending, p_ranging, p_volatile
        config: RegimeTSMOMConfig 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    # 입력 검증
    required_cols = {
        "vw_momentum",
        "realized_vol",
        "p_trending",
        "p_ranging",
        "p_volatile",
    }
    if config.short_mode == ShortMode.HEDGE_ONLY:
        required_cols.add("drawdown")

    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 컬럼 추출
    momentum: pd.Series = df["vw_momentum"]  # type: ignore[assignment]
    realized_vol: pd.Series = df["realized_vol"]  # type: ignore[assignment]
    p_trending: pd.Series = df["p_trending"]  # type: ignore[assignment]
    p_ranging: pd.Series = df["p_ranging"]  # type: ignore[assignment]
    p_volatile: pd.Series = df["p_volatile"]  # type: ignore[assignment]

    # 1. 모멘텀 방향
    momentum_direction = np.sign(momentum)

    # 2. Probability-weighted adaptive vol_target (단일 채널)
    adaptive_vol_target: pd.Series = (  # type: ignore[assignment]
        p_trending * config.trending_vol_target
        + p_ranging * config.ranging_vol_target
        + p_volatile * config.volatile_vol_target
    )

    # 3. Vol scalar = adaptive_vol_target / realized_vol
    clamped_vol = realized_vol.clip(lower=config.min_volatility)
    vol_scalar: pd.Series = adaptive_vol_target / clamped_vol  # type: ignore[assignment]

    # 4. Raw strength = direction * vol_scalar (leverage_scale 이중 곱셈 제거)
    raw_strength = momentum_direction * vol_scalar

    # 5. Shift(1): 미래 참조 편향 방지
    signal_shifted: pd.Series = raw_strength.shift(1)  # type: ignore[assignment]

    # 6. Direction
    direction_raw = pd.Series(np.sign(signal_shifted), index=df.index)
    direction = pd.Series(
        direction_raw.fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    # 7. Strength
    strength = pd.Series(
        signal_shifted.fillna(0),
        index=df.index,
        name="strength",
    )

    # 8. 숏 모드 처리
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

    # 9. Entry/Exit
    prev_direction = direction.shift(1).fillna(0)
    long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)
    short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)
    entries = pd.Series(long_entry | short_entry, index=df.index, name="entries")

    to_neutral = (direction == Direction.NEUTRAL) & (prev_direction != Direction.NEUTRAL)
    reversal = direction * prev_direction < 0
    exits = pd.Series(to_neutral | reversal, index=df.index, name="exits")

    # 로깅
    valid_strength = strength[strength != 0]
    if len(valid_strength) > 0:
        valid_regime = df["regime_label"].dropna()
        if len(valid_regime) > 0:
            regime_counts = valid_regime.value_counts()
            regime_str = ", ".join(f"{k}: {v}" for k, v in regime_counts.items())
            logger.info("🏷️ Regime Distribution | {}", regime_str)

        avg_vol_target = adaptive_vol_target.dropna().mean()
        logger.info(
            "📊 Adaptive Params | Avg Vol Target: {:.2f}",
            avg_vol_target,
        )

    return StrategySignals(
        entries=entries,
        exits=exits,
        direction=direction,
        strength=strength,
    )
