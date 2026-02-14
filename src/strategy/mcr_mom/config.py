"""Momentum Crash Filter strategy configuration.

Standard momentum + VoV-based crash filter.
VoV-Mom lesson: use VoV as defensive override only, not as alpha source.
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


class McrMomConfig(BaseModel):
    """Momentum Crash Filter strategy configuration.

    Attributes:
        mom_lookback: Momentum return lookback (bars).
        vov_window: Volatility-of-volatility calculation window.
        vov_crash_threshold: VoV percentile above which crash filter activates.
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
    mom_lookback: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Momentum return lookback (bars).",
    )
    vov_window: int = Field(
        default=20,
        ge=5,
        le=60,
        description="VoV calculation window.",
    )
    vov_crash_threshold: float = Field(
        default=0.8,
        ge=0.5,
        le=0.99,
        description="VoV percentile rank above which crash filter activates.",
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
        # vol_window for realized_vol, then vov_window on top, plus rolling rank
        return max(self.mom_lookback, self.vol_window + self.vov_window, self.atr_period) + 60 + 10
