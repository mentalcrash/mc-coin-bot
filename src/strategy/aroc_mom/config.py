"""Adaptive ROC Momentum strategy configuration.

Dynamically adjust momentum lookback based on volatility regime:
high volatility -> shorter lookback, low volatility -> longer lookback.
Barroso & Santa-Clara (2015) dynamic risk management extension.
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


class ArocMomConfig(BaseModel):
    """Adaptive ROC Momentum strategy configuration.

    Attributes:
        fast_lookback: Minimum ROC lookback (high-vol regime).
        slow_lookback: Maximum ROC lookback (low-vol regime).
        vol_rank_window: Volatility percentile rank window.
        mom_threshold: Minimum absolute ROC for entry.
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
    fast_lookback: int = Field(
        default=10,
        ge=3,
        le=60,
        description="Minimum ROC lookback (high-vol regime).",
    )
    slow_lookback: int = Field(
        default=60,
        ge=20,
        le=200,
        description="Maximum ROC lookback (low-vol regime).",
    )
    vol_rank_window: int = Field(
        default=60,
        ge=20,
        le=252,
        description="Volatility percentile rank lookback.",
    )
    mom_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=0.2,
        description="Minimum absolute ROC for entry.",
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
        if self.slow_lookback <= self.fast_lookback:
            msg = (
                f"slow_lookback ({self.slow_lookback}) must be > "
                f"fast_lookback ({self.fast_lookback})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """Required warmup period (in bars)."""
        return max(self.slow_lookback, self.vol_rank_window, self.vol_window, self.atr_period) + 10
