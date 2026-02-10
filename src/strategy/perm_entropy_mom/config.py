"""Permutation Entropy Momentum Strategy Configuration.

Low PE (orderly market) = high momentum conviction.
High PE (noisy market) = reduced position / neutral.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """Short position handling mode.

    Attributes:
        DISABLED: Long-Only mode (short signal -> neutral)
        HEDGE_ONLY: Hedge-purpose short only (above drawdown threshold)
        FULL: Full Long/Short mode
    """

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class PermEntropyMomConfig(BaseModel):
    """Permutation Entropy Momentum strategy configuration.

    Low Permutation Entropy indicates orderly price patterns (trending),
    while high PE indicates random/noisy price movement.

    Signal Formula:
        1. PE = -sum(p_i * log(p_i)) / log(m!) -> [0, 1]
        2. conviction_scaler = 1 - PE_normalized
        3. momentum_direction = sign(rolling_return(lookback))
        4. raw_weight = direction * vol_target / realized_vol
        5. final_weight = raw_weight * conviction_scaler (shift(1))
        6. Noise gate: PE > noise_threshold -> weight = 0

    Attributes:
        pe_order: Permutation entropy order (m). Higher m = more granular patterns.
        pe_short_window: Short-term PE rolling window (bars).
        pe_long_window: Long-term PE rolling window (bars).
        mom_lookback: Momentum direction lookback (bars).
        noise_threshold: PE threshold above which signal is zeroed (pure noise zone).
        vol_target: Annual target volatility.
        min_volatility: Minimum volatility clamp (prevents division by zero).
        annualization_factor: Annualization factor (4H bars: 6*365 = 2190).
        use_log_returns: Whether to use log returns.
        atr_period: ATR calculation period (for trailing stop).
        short_mode: Short position handling mode.
        hedge_threshold: Hedge short activation drawdown threshold.
        hedge_strength_ratio: Hedge short strength ratio (vs long).
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # Permutation Entropy parameters
    # =========================================================================
    pe_order: int = Field(
        default=3,
        ge=2,
        le=7,
        description="Permutation entropy order (m). 2~7, default 3.",
    )
    pe_short_window: int = Field(
        default=30,
        ge=10,
        le=120,
        description="Short-term PE rolling window (bars). ~5d at 4H.",
    )
    pe_long_window: int = Field(
        default=60,
        ge=20,
        le=240,
        description="Long-term PE rolling window (bars). ~10d at 4H.",
    )

    # =========================================================================
    # Momentum parameters
    # =========================================================================
    mom_lookback: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Momentum direction lookback (bars).",
    )
    noise_threshold: float = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        description="PE above this = pure noise zone -> weight = 0.",
    )

    # =========================================================================
    # Volatility common parameters
    # =========================================================================
    vol_target: float = Field(
        default=0.35,
        ge=0.05,
        le=1.0,
        description="Annual target volatility.",
    )
    min_volatility: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="Minimum volatility clamp (prevents division by zero).",
    )
    annualization_factor: float = Field(
        default=2190.0,
        gt=0,
        description="Annualization factor (4H bars: 6*365 = 2190).",
    )

    # =========================================================================
    # Options
    # =========================================================================
    use_log_returns: bool = Field(
        default=True,
        description="Whether to use log returns (recommended: True).",
    )
    atr_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ATR calculation period (for trailing stop).",
    )

    # =========================================================================
    # Short mode settings
    # =========================================================================
    short_mode: ShortMode = Field(
        default=ShortMode.HEDGE_ONLY,
        description="Short position handling mode (DISABLED/HEDGE_ONLY/FULL).",
    )
    hedge_threshold: float = Field(
        default=-0.07,
        ge=-0.30,
        le=-0.05,
        description="Hedge short activation drawdown threshold (e.g. -0.07 = -7%).",
    )
    hedge_strength_ratio: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="Hedge short strength ratio (vs long, e.g. 0.8 = 80%).",
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """Configuration consistency validation."""
        if self.pe_long_window <= self.pe_short_window:
            msg = (
                f"pe_long_window ({self.pe_long_window}) must be > "
                f"pe_short_window ({self.pe_short_window})"
            )
            raise ValueError(msg)

        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """Required warmup period (in bars)."""
        # pe_long_window is the largest rolling window + pe_order for stride
        return max(self.pe_long_window + self.pe_order, self.mom_lookback, self.atr_period) + 1
