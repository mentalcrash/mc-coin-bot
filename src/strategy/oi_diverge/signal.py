"""OI-Price Divergence Signal Generation.

Detection:
    - Short squeeze: OI-price divergence + extreme negative FR → LONG
    - Long liquidation: OI-price divergence + extreme positive FR → SHORT

Strength = direction * vol_scalar * clip(|fr_zscore| / threshold, 1, 2)
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.models.types import Direction
from src.strategy.oi_diverge.config import OiDivergeConfig
from src.strategy.tsmom.config import ShortMode
from src.strategy.types import StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: OiDivergeConfig | None = None,
) -> StrategySignals:
    """스퀴즈/청산 시그널 생성.

    Args:
        df: preprocess() 결과 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    if config is None:
        config = OiDivergeConfig()

    required = {"oi_price_div", "fr_zscore", "vol_scalar"}
    missing = required - set(df.columns)
    if missing:
        msg = f"Missing columns: {missing}"
        raise ValueError(msg)

    oi_div: pd.Series = df["oi_price_div"]  # type: ignore[assignment]
    fr_z: pd.Series = df["fr_zscore"]  # type: ignore[assignment]
    vol_scalar: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    div_thresh = config.divergence_threshold
    fr_thresh = config.fr_zscore_threshold

    # Divergence detected (negative correlation = price-OI diverging)
    is_diverging = oi_div < div_thresh

    # Short squeeze: divergence + extreme negative FR (shorts crowded)
    short_squeeze = is_diverging & (fr_z < -fr_thresh)

    # Long liquidation: divergence + extreme positive FR (longs crowded)
    long_liq = is_diverging & (fr_z > fr_thresh)

    # Direction
    raw_dir = pd.Series(
        np.where(short_squeeze, 1.0, np.where(long_liq, -1.0, 0.0)),
        index=df.index,
    )

    # Strength: magnitude scales with |fr_zscore| / threshold (clamped 1~2)
    fr_magnitude = np.clip(fr_z.abs() / max(fr_thresh, 1e-8), 1.0, 2.0)
    raw_strength = raw_dir * vol_scalar * fr_magnitude

    # Shift(1) — lookahead bias 방지
    signal_shifted = raw_strength.shift(1).fillna(0.0)

    direction = pd.Series(
        np.sign(signal_shifted).fillna(0).astype(int),
        index=df.index,
        name="direction",
    )
    strength = pd.Series(signal_shifted, index=df.index, name="strength")

    # ShortMode 처리
    if config.short_mode == ShortMode.DISABLED:
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)
    elif config.short_mode == ShortMode.HEDGE_ONLY:
        # OI-diverge는 기본 FULL이므로 이 경우는 드뭄
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    # Entry / Exit
    prev_dir = direction.shift(1).fillna(0)
    long_entry = (direction == Direction.LONG) & (prev_dir != Direction.LONG)
    short_entry = (direction == Direction.SHORT) & (prev_dir != Direction.SHORT)
    entries = pd.Series(long_entry | short_entry, index=df.index, name="entries")

    to_neutral = (direction == Direction.NEUTRAL) & (prev_dir != Direction.NEUTRAL)
    reversal = direction * prev_dir < 0
    exits = pd.Series(to_neutral | reversal, index=df.index, name="exits")

    # Stats
    n_total = len(df)
    n_long = int((direction == Direction.LONG).sum())
    n_short = int((direction == Direction.SHORT).sum())
    n_squeeze = int(short_squeeze.sum())
    n_liq = int(long_liq.sum())
    logger.info(
        "OI-Diverge signals | Total: {}, Long: {} ({:.1%}), Short: {} ({:.1%}), Squeeze: {}, Liq: {}",
        n_total,
        n_long,
        n_long / max(n_total, 1),
        n_short,
        n_short / max(n_total, 1),
        n_squeeze,
        n_liq,
    )

    return StrategySignals(entries=entries, exits=exits, direction=direction, strength=strength)
