"""On-chain Accumulation Signal Generation.

2/3 majority vote:
    - MVRV score: undervalued (+1), overvalued (-1), neutral (0)
    - Flow score: outflow (+1, accumulation), inflow (-1, distribution)
    - Stablecoin score: increasing (+1, dry powder), decreasing (-1)

Composite >= 2 → LONG, <= -2 → EXIT (short_mode DISABLED)
Strength = direction * vol_scalar * (|composite| / 3.0)
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.models.types import Direction
from src.strategy.onchain_accum.config import OnchainAccumConfig
from src.strategy.tsmom.config import ShortMode
from src.strategy.types import StrategySignals

# Minimum votes for consensus (2 out of 3)
_CONSENSUS_THRESHOLD = 2
_NUM_INDICATORS = 3.0


def generate_signals(
    df: pd.DataFrame,
    config: OnchainAccumConfig | None = None,
) -> StrategySignals:
    """2/3 다수결 시그널 생성.

    Args:
        df: preprocess() 결과 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    if config is None:
        config = OnchainAccumConfig()

    required = {"vol_scalar"}
    missing = required - set(df.columns)
    if missing:
        msg = f"Missing columns: {missing}"
        raise ValueError(msg)

    vol_scalar: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    # 1. MVRV score
    if "oc_mvrv" in df.columns:
        mvrv: pd.Series = df["oc_mvrv"]  # type: ignore[assignment]
        mvrv_score = pd.Series(
            np.where(
                mvrv < config.mvrv_undervalued,
                1,
                np.where(mvrv > config.mvrv_overvalued, -1, 0),
            ),
            index=df.index,
            dtype=float,
        )
    else:
        mvrv_score = pd.Series(0.0, index=df.index)

    # 2. Flow score (negative zscore = outflow = accumulation → bullish)
    if "net_flow_zscore" in df.columns:
        nf_z: pd.Series = df["net_flow_zscore"]  # type: ignore[assignment]
        flow_score = pd.Series(
            np.where(
                nf_z < -config.flow_threshold,
                1,
                np.where(nf_z > config.flow_threshold, -1, 0),
            ),
            index=df.index,
            dtype=float,
        )
    else:
        flow_score = pd.Series(0.0, index=df.index)

    # 3. Stablecoin score
    if "stablecoin_roc" in df.columns:
        stab_roc: pd.Series = df["stablecoin_roc"]  # type: ignore[assignment]
        stab_score = pd.Series(
            np.where(
                stab_roc > config.stablecoin_roc_threshold,
                1,
                np.where(stab_roc < -config.stablecoin_roc_threshold, -1, 0),
            ),
            index=df.index,
            dtype=float,
        )
    else:
        stab_score = pd.Series(0.0, index=df.index)

    # NaN → 0 (neutral)
    mvrv_score = mvrv_score.fillna(0.0)
    flow_score = flow_score.fillna(0.0)
    stab_score = stab_score.fillna(0.0)

    # Composite: [-3, +3]
    composite = mvrv_score + flow_score + stab_score

    # Direction: 2/3 consensus required
    raw_dir = pd.Series(
        np.where(
            composite >= _CONSENSUS_THRESHOLD,
            1.0,
            np.where(composite <= -_CONSENSUS_THRESHOLD, -1.0, 0.0),
        ),
        index=df.index,
    )

    # Strength: proportional to consensus strength
    magnitude = composite.abs() / _NUM_INDICATORS
    raw_strength = raw_dir * vol_scalar * magnitude

    # Shift(1) — lookahead bias 방지
    signal_shifted = raw_strength.shift(1).fillna(0.0)

    direction = pd.Series(
        np.sign(signal_shifted).fillna(0).astype(int),
        index=df.index,
        name="direction",
    )
    strength = pd.Series(signal_shifted, index=df.index, name="strength")

    # ShortMode: DISABLED → suppress all shorts
    if config.short_mode in {ShortMode.DISABLED, ShortMode.HEDGE_ONLY}:
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
    n_bullish = int((composite >= _CONSENSUS_THRESHOLD).sum())
    n_bearish = int((composite <= -_CONSENSUS_THRESHOLD).sum())
    logger.info(
        "Onchain-Accum signals | Total: {}, Long: {} ({:.1%}), Bullish: {}, Bearish: {}",
        n_total,
        n_long,
        n_long / max(n_total, 1),
        n_bullish,
        n_bearish,
    )

    return StrategySignals(entries=entries, exits=exits, direction=direction, strength=strength)
