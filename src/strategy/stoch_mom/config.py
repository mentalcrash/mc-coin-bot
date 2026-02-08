"""Stochastic Momentum Hybrid Strategy Configuration.

Stochastic %K/%D crossover + SMA(30) trend filter + ATR dynamic position sizing.

Signal Logic:
    - Long: %K crosses above %D AND close > SMA(30)
    - Short: %K crosses below %D AND close < SMA(30)
    - Exit: opposite crossover
    - strength = direction * vol_scalar * vol_ratio

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """Short position handling mode.

    Attributes:
        DISABLED: Long-Only mode (short signals -> neutral)
        FULL: Full Long/Short mode
    """

    DISABLED = 0
    FULL = 2


class StochMomConfig(BaseModel):
    """Stochastic Momentum Hybrid strategy configuration.

    Stochastic %K/%D crossover + SMA trend filter + ATR-based dynamic
    position sizing. Combines mean-reversion oscillator signals with
    trend confirmation and volatility-adjusted sizing.

    Signal Formula:
        1. Stochastic %K/%D crossover (shifted to avoid lookahead)
        2. SMA trend filter (close > SMA for long, close < SMA for short)
        3. vol_scalar = vol_target / realized_vol
        4. vol_ratio = (atr / close), clipped to [min_vol_ratio, max_vol_ratio]
        5. strength = direction * vol_scalar * vol_ratio

    Attributes:
        k_period: Stochastic %K period
        d_period: %D smoothing period (SMA of %K)
        sma_period: Trend filter SMA period
        atr_period: ATR calculation period
        vol_target: Annual target volatility (0.0~1.0)
        min_volatility: Minimum volatility clamp (division-by-zero guard)
        min_vol_ratio: Minimum position size ratio
        max_vol_ratio: Maximum position size ratio
        annualization_factor: Annualization factor (daily: 365)
        short_mode: Short position handling mode (DISABLED/FULL)

    Example:
        >>> config = StochMomConfig(
        ...     k_period=14,
        ...     d_period=3,
        ...     sma_period=30,
        ...     vol_target=0.40,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # Stochastic parameters
    k_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="Stochastic %K period",
    )
    d_period: int = Field(
        default=3,
        ge=2,
        le=10,
        description="%D smoothing period (SMA of %K)",
    )

    # Trend filter
    sma_period: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Trend filter SMA period",
    )

    # ATR
    atr_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ATR calculation period",
    )

    # Volatility / Position sizing
    vol_target: float = Field(
        default=0.40,
        ge=0.05,
        le=1.0,
        description="Annual target volatility (0.40 = 40%)",
    )
    min_volatility: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="Minimum volatility clamp (division-by-zero guard)",
    )
    min_vol_ratio: float = Field(
        default=0.30,
        ge=0.1,
        le=0.8,
        description="Minimum position size ratio from ATR/close",
    )
    max_vol_ratio: float = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        description="Maximum position size ratio from ATR/close",
    )

    # Time frame
    annualization_factor: float = Field(
        default=365.0,
        gt=0,
        description="Annualization factor (daily: 365, 4h: 2190, 1h: 8760)",
    )

    # Short mode
    short_mode: ShortMode = Field(
        default=ShortMode.DISABLED,
        description="Short position handling mode (DISABLED=0, FULL=2)",
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """Configuration consistency validation.

        Validation rules:
            - vol_target >= min_volatility
            - min_vol_ratio < max_vol_ratio

        Returns:
            Validated self

        Raises:
            ValueError: When configuration is inconsistent
        """
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) should be >= "
                f"min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        if self.min_vol_ratio >= self.max_vol_ratio:
            msg = (
                f"min_vol_ratio ({self.min_vol_ratio}) must be < "
                f"max_vol_ratio ({self.max_vol_ratio})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """Required warmup period (number of candles).

        The longest rolling window is max(k_period, sma_period, atr_period),
        plus d_period for %D smoothing, plus 1 for shift.

        Returns:
            Required number of candles
        """
        return max(self.k_period, self.sma_period, self.atr_period) + self.d_period + 1

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> StochMomConfig:
        """Create default configuration for a specific timeframe.

        Args:
            timeframe: Timeframe string (e.g., "1h", "4h", "1d")
            **kwargs: Additional setting overrides

        Returns:
            StochMomConfig optimized for the given timeframe
        """
        annualization_map: dict[str, float] = {
            "1m": 525600.0,
            "5m": 105120.0,
            "15m": 35040.0,
            "1h": 8760.0,
            "4h": 2190.0,
            "1d": 365.0,
        }

        annualization = annualization_map.get(timeframe, 365.0)

        return cls(
            annualization_factor=annualization,
            **kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def conservative(cls) -> StochMomConfig:
        """Conservative preset (longer periods, lower vol target).

        Returns:
            StochMomConfig with conservative parameters
        """
        return cls(
            k_period=21,
            sma_period=50,
            vol_target=0.30,
            min_volatility=0.08,
        )

    @classmethod
    def aggressive(cls) -> StochMomConfig:
        """Aggressive preset (shorter periods, higher vol target).

        Returns:
            StochMomConfig with aggressive parameters
        """
        return cls(
            k_period=9,
            sma_period=20,
            vol_target=0.50,
            min_volatility=0.05,
        )
