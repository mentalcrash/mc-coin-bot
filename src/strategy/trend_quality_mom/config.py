"""Trend Quality Momentum strategy configuration.

R-squared of linear regression measures trend quality (0~1).
sign(slope) * R^2 gives directional conviction for momentum.
High R^2 = orderly trend = high conviction. Low R^2 = noise = reduced position.
"""

from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """Short position handling mode."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class TrendQualityMomConfig(BaseModel):
    """Trend Quality Momentum strategy configuration.

    Signal Formula:
        1. slope, r_squared = OLS(close, time_index) over lookback window
        2. direction = sign(slope) where R^2 > r2_threshold, else 0
        3. conviction = R^2 (continuous, 0~1)
        4. strength = direction * vol_scalar * conviction

    Attributes:
        regression_lookback: Linear regression lookback window (bars).
        r2_threshold: Minimum R^2 to generate signal (noise gate).
        mom_lookback: Momentum return lookback for direction confirmation.
        vol_target: Annual target volatility.
        vol_window: Volatility calculation rolling window.
        min_volatility: Minimum volatility clamp.
        annualization_factor: 4H = 2190.0.
        atr_period: ATR calculation period (trailing stop).
        short_mode: Short position handling mode.
        hedge_threshold: Hedge activation drawdown threshold.
        hedge_strength_ratio: Hedge short strength scaling.
    """

    model_config = ConfigDict(frozen=True)

    # --- Trend Quality Parameters ---
    regression_lookback: int = Field(
        default=30,
        ge=10,
        le=120,
        description="Linear regression lookback window (bars).",
    )
    r2_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=0.9,
        description="Minimum R^2 to generate signal (noise gate).",
    )
    mom_lookback: int = Field(
        default=20,
        ge=5,
        le=120,
        description="Momentum return lookback for direction confirmation.",
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
            max(self.regression_lookback, self.mom_lookback, self.vol_window, self.atr_period) + 10
        )
