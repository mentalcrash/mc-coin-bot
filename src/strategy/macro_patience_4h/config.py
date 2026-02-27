"""Macro-Gated Patient Trend (4H) — configuration."""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """Short-selling mode."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class MacroPatience4hConfig(BaseModel):
    """Config for Macro-Gated Patient Trend strategy.

    Macro(DXY/VIX/M2) z-score composite determines direction.
    4H multi-scale Donchian channel provides entry timing.
    Macro gate reduces trade frequency to ~25-40/year.
    """

    model_config = ConfigDict(frozen=True)

    # --- Macro gate parameters ---
    macro_z_window: int = Field(default=60, ge=20, le=252)
    macro_z_threshold: float = Field(default=0.3, ge=0.0, le=2.0)
    dxy_weight: float = Field(default=-1.0, ge=-3.0, le=3.0)
    vix_weight: float = Field(default=-1.0, ge=-3.0, le=3.0)
    m2_weight: float = Field(default=1.0, ge=-3.0, le=3.0)
    m2_growth_window: int = Field(default=60, ge=20, le=252)

    # --- Multi-scale Donchian parameters (4H bars) ---
    dc_scale_short: int = Field(default=90, ge=30, le=200)
    dc_scale_mid: int = Field(default=180, ge=60, le=400)
    dc_scale_long: int = Field(default=360, ge=120, le=800)
    entry_threshold: float = Field(default=0.34, ge=0.0, le=1.0)

    # --- Vol-target (common) ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=2190.0, gt=0.0)

    # --- Short mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_scales(self) -> MacroPatience4hConfig:
        if not (self.dc_scale_short < self.dc_scale_mid < self.dc_scale_long):
            msg = (
                f"Scales must be ascending: "
                f"{self.dc_scale_short} < {self.dc_scale_mid} < {self.dc_scale_long}"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        return self.dc_scale_long + self.macro_z_window + 10
