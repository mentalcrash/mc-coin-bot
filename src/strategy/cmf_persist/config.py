"""CMF Trend Persistence strategy configuration.

Chaikin Money Flow sign persistence over N bars captures institutional
accumulation/distribution patterns.
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


class CmfPersistConfig(BaseModel):
    """CMF Trend Persistence strategy configuration.

    Attributes:
        cmf_period: CMF calculation period.
        persist_window: Consecutive bar count window.
        persist_threshold: Minimum positive-bar ratio for signal.
        cmf_strength_clip: Maximum |CMF| for conviction scaling.
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
    cmf_period: int = Field(
        default=20,
        ge=5,
        le=60,
        description="CMF calculation period.",
    )
    persist_window: int = Field(
        default=10,
        ge=3,
        le=40,
        description="Number of bars to check persistence.",
    )
    persist_threshold: float = Field(
        default=0.7,
        ge=0.5,
        le=1.0,
        description="Minimum positive-bar ratio for signal (0.7 = 7/10).",
    )
    cmf_strength_clip: float = Field(
        default=0.3,
        ge=0.05,
        le=1.0,
        description="Maximum |CMF| for conviction scaling.",
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
        return self

    def warmup_periods(self) -> int:
        """Required warmup period (in bars)."""
        return max(self.cmf_period + self.persist_window, self.vol_window, self.atr_period) + 10
