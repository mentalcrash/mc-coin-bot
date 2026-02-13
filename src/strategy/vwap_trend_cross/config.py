"""VWAP Trend Crossover strategy configuration.

Short/long rolling VWAP crossover detects shifts in average participant entry price.
Crypto whale trading concentration makes VWAP shifts a meaningful trend signal.
"""

from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """Short position handling mode."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class VwapTrendCrossConfig(BaseModel):
    """VWAP Trend Crossover strategy configuration.

    Signal Formula:
        1. vwap = cumsum(close * volume) / cumsum(volume) over rolling window
        2. vwap_short = rolling VWAP(short_window)
        3. vwap_long = rolling VWAP(long_window)
        4. long when vwap_short > vwap_long AND close > vwap_short
        5. short when vwap_short < vwap_long AND close < vwap_short
        6. strength = direction * vol_scalar * vwap_spread (conviction)

    Attributes:
        vwap_short_window: Short-term rolling VWAP window (bars).
        vwap_long_window: Long-term rolling VWAP window (bars).
        spread_clip: Maximum VWAP spread for conviction scaling.
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

    # --- VWAP Parameters ---
    vwap_short_window: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Short-term rolling VWAP window (bars).",
    )
    vwap_long_window: int = Field(
        default=60,
        ge=20,
        le=300,
        description="Long-term rolling VWAP window (bars).",
    )
    spread_clip: float = Field(
        default=0.05,
        ge=0.005,
        le=0.20,
        description="Max VWAP spread (as fraction of price) for conviction clipping.",
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
        if self.vwap_long_window <= self.vwap_short_window:
            msg = (
                f"vwap_long_window ({self.vwap_long_window}) must be > "
                f"vwap_short_window ({self.vwap_short_window})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """Required warmup period (in bars)."""
        return max(self.vwap_long_window, self.vol_window, self.atr_period) + 10
