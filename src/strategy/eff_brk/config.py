"""Efficiency Breakout strategy configuration.

Kaufman Efficiency Ratio threshold breakout to detect noise-to-trend transitions.
ER as PRIMARY breakout detector.
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


class EffBrkConfig(BaseModel):
    """Efficiency Breakout strategy configuration.

    Attributes:
        er_period: Efficiency Ratio calculation period.
        er_threshold: ER breakout threshold (trend detected when ER > threshold).
        er_exit_threshold: ER exit threshold (exit when ER drops below).
        mom_lookback: Momentum direction lookback (bars).
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
    er_period: int = Field(
        default=10,
        ge=3,
        le=60,
        description="Efficiency Ratio calculation period.",
    )
    er_threshold: float = Field(
        default=0.35,
        ge=0.1,
        le=0.9,
        description="ER breakout threshold.",
    )
    er_exit_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="ER exit threshold.",
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
        if self.er_threshold <= self.er_exit_threshold:
            msg = (
                f"er_threshold ({self.er_threshold}) must be > "
                f"er_exit_threshold ({self.er_exit_threshold})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """Required warmup period (in bars)."""
        return max(self.er_period, self.mom_lookback, self.vol_window, self.atr_period) + 10
