"""Volume-Confirmed Momentum strategy configuration.

Momentum + volume trend confirmation: enter only when both price momentum
and volume trend agree. Volume rising = participation increasing = momentum fuel.
"""

from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """Short position handling mode."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class VolConfirmMomConfig(BaseModel):
    """Volume-Confirmed Momentum strategy configuration.

    Signal Formula:
        1. momentum = sign(rolling_return(mom_lookback))
        2. vol_sma_short = SMA(volume, vol_short_window)
        3. vol_sma_long = SMA(volume, vol_long_window)
        4. vol_rising = vol_sma_short > vol_sma_long
        5. entry = momentum direction & vol_rising (both agree)
        6. strength = direction * vol_scalar * vol_ratio_conviction

    Attributes:
        mom_lookback: Momentum return lookback (bars).
        vol_short_window: Short-term volume SMA window.
        vol_long_window: Long-term volume SMA window.
        vol_ratio_clip: Maximum vol ratio for conviction scaling.
        vol_target: Annual target volatility.
        vol_window: Volatility calculation rolling window.
        min_volatility: Minimum volatility clamp.
        annualization_factor: 4H = 2190.0.
        atr_period: ATR calculation period.
        short_mode: Short position handling mode.
        hedge_threshold: Hedge activation drawdown threshold.
        hedge_strength_ratio: Hedge short strength scaling.
    """

    model_config = ConfigDict(frozen=True)

    # --- Momentum Parameters ---
    mom_lookback: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Momentum return lookback (bars).",
    )

    # --- Volume Confirmation Parameters ---
    vol_short_window: int = Field(
        default=10,
        ge=3,
        le=60,
        description="Short-term volume SMA window.",
    )
    vol_long_window: int = Field(
        default=40,
        ge=10,
        le=200,
        description="Long-term volume SMA window.",
    )
    vol_ratio_clip: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="Max vol ratio for conviction scaling (clip upper bound).",
    )

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=2190.0, gt=0)

    # --- Options ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> Self:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.vol_long_window <= self.vol_short_window:
            msg = (
                f"vol_long_window ({self.vol_long_window}) must be > "
                f"vol_short_window ({self.vol_short_window})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """Required warmup period (in bars)."""
        return max(self.mom_lookback, self.vol_long_window, self.vol_window, self.atr_period) + 10
