"""Return Persistence Score strategy configuration.

Rolling ratio of positive return bars captures trend persistence.
Extreme simplicity to minimize overfitting risk.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """Short position handling mode."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class RetPersistConfig(BaseModel):
    """Return Persistence Score strategy configuration.

    Attributes:
        persist_window: Rolling window for positive return bar ratio.
        long_threshold: Positive ratio above which to go long.
        short_threshold: Positive ratio below which to go short.
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

    # --- Strategy Parameters ---
    persist_window: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Rolling window for positive return bar ratio.",
    )
    long_threshold: float = Field(
        default=0.6,
        ge=0.5,
        le=0.9,
        description="Positive ratio above which to go long.",
    )
    short_threshold: float = Field(
        default=0.4,
        ge=0.1,
        le=0.5,
        description="Positive ratio below which to go short.",
    )

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=2190.0, gt=0)

    # --- Options ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> Self:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.long_threshold <= self.short_threshold:
            msg = (
                f"long_threshold ({self.long_threshold}) must be > "
                f"short_threshold ({self.short_threshold})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """Required warmup period (in bars)."""
        return max(self.persist_window, self.vol_window, self.atr_period) + 10
