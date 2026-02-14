"""Dual Volatility Trend strategy configuration.

Yang-Zhang vs Parkinson volatility ratio to distinguish information arrival
from noise. Cross-estimator comparison is novel across the strategy universe.
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


class DualVolConfig(BaseModel):
    """Dual Volatility Trend strategy configuration.

    Attributes:
        vol_estimator_window: Window for YZ and Parkinson vol estimators.
        ratio_smooth: EMA smoothing period for vol ratio.
        ratio_upper: Upper threshold (info arrival > noise -> trend).
        ratio_lower: Lower threshold (noise dominant -> no trade).
        mom_lookback: Momentum direction lookback.
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
    vol_estimator_window: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Window for YZ and Parkinson vol estimators.",
    )
    ratio_smooth: int = Field(
        default=10,
        ge=3,
        le=50,
        description="EMA smoothing for vol ratio.",
    )
    ratio_upper: float = Field(
        default=1.2,
        ge=0.8,
        le=3.0,
        description="Upper ratio threshold (trend regime).",
    )
    ratio_lower: float = Field(
        default=0.8,
        ge=0.3,
        le=1.5,
        description="Lower ratio threshold (noise regime).",
    )
    mom_lookback: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Momentum direction lookback (bars).",
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
        if self.ratio_upper <= self.ratio_lower:
            msg = f"ratio_upper ({self.ratio_upper}) must be > ratio_lower ({self.ratio_lower})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """Required warmup period (in bars)."""
        return (
            max(self.vol_estimator_window, self.mom_lookback, self.vol_window, self.atr_period)
            + self.ratio_smooth
            + 10
        )
