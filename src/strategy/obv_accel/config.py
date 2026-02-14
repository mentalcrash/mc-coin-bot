"""OBV Acceleration Momentum strategy configuration.

OBV second derivative (acceleration) captures changes in smart money activity intensity.
CTREND uses normalized OBV (1st derivative); this uses 2nd derivative.
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


class ObvAccelConfig(BaseModel):
    """OBV Acceleration Momentum strategy configuration.

    Attributes:
        obv_smooth: EMA smoothing for OBV (reduces noise).
        accel_window: Window for 2nd derivative (acceleration).
        accel_threshold: Minimum |acceleration| z-score for signal.
        mom_lookback: Price momentum lookback for direction confirmation.
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
    obv_smooth: int = Field(
        default=10,
        ge=3,
        le=50,
        description="EMA smoothing period for OBV.",
    )
    accel_window: int = Field(
        default=10,
        ge=3,
        le=40,
        description="Window for acceleration (2nd derivative) calculation.",
    )
    accel_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=3.0,
        description="Minimum |acceleration| z-score for signal.",
    )
    mom_lookback: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Price momentum lookback for direction.",
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
        return (
            max(
                self.obv_smooth + self.accel_window,
                self.mom_lookback,
                self.vol_window,
                self.atr_period,
            )
            + 30
            + 10
        )
