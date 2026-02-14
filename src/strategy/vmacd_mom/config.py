"""Volume MACD Momentum strategy configuration.

Volume MACD as PRIMARY signal + price momentum confirmation.
VMACD > 0 + positive price momentum = volume-backed trend.
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


class VmacdMomConfig(BaseModel):
    """Volume MACD Momentum strategy configuration.

    Attributes:
        vmacd_fast: Volume MACD fast EMA period.
        vmacd_slow: Volume MACD slow EMA period.
        vmacd_signal: Volume MACD signal line EMA period.
        mom_lookback: Price momentum lookback (bars).
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
    vmacd_fast: int = Field(
        default=12,
        ge=3,
        le=30,
        description="Volume MACD fast EMA period.",
    )
    vmacd_slow: int = Field(
        default=26,
        ge=10,
        le=60,
        description="Volume MACD slow EMA period.",
    )
    vmacd_signal: int = Field(
        default=9,
        ge=3,
        le=30,
        description="Volume MACD signal line EMA period.",
    )
    mom_lookback: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Price momentum lookback (bars).",
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
        if self.vmacd_slow <= self.vmacd_fast:
            msg = f"vmacd_slow ({self.vmacd_slow}) must be > vmacd_fast ({self.vmacd_fast})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """Required warmup period (in bars)."""
        return (
            max(
                self.vmacd_slow + self.vmacd_signal,
                self.mom_lookback,
                self.vol_window,
                self.atr_period,
            )
            + 10
        )
