"""HMM Regime Strategy Configuration."""

from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """Short position handling mode.

    Attributes:
        DISABLED: Long-Only mode (short signals -> neutral)
        HEDGE_ONLY: Hedge-purpose shorts only (when drawdown exceeds threshold)
        FULL: Full Long/Short mode
    """

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class HMMRegimeConfig(BaseModel):
    """HMM Regime Strategy Configuration.

    GaussianHMM 기반 3-state (Bull/Bear/Sideways) regime classification.
    Expanding window training으로 look-ahead bias 방지.

    Signal Formula:
        1. HMM expanding window training → regime (Bull/Bear/Sideways)
        2. direction = regime label (+1, -1, 0)
        3. strength = direction * regime_prob * vol_scalar

    Attributes:
        n_states: Number of HMM hidden states
        min_train_window: Minimum expanding window for HMM training
        retrain_interval: Retrain frequency in bars
        vol_window: Rolling volatility window
        vol_target: Annual volatility target
        min_volatility: Minimum volatility clamp
        annualization_factor: Annualization factor
        use_log_returns: Whether to use log returns
        atr_period: ATR calculation period
        n_iter: HMM training iterations
        short_mode: Short position handling mode
        hedge_threshold: Hedge short activation drawdown threshold
        hedge_strength_ratio: Hedge short strength ratio
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # HMM Parameters
    # =========================================================================
    n_states: int = Field(
        default=3,
        ge=2,
        le=5,
        description="Number of HMM hidden states",
    )
    min_train_window: int = Field(
        default=252,
        ge=100,
        le=500,
        description="Minimum expanding window for HMM training",
    )
    retrain_interval: int = Field(
        default=21,
        ge=1,
        le=63,
        description="Retrain frequency in bars",
    )
    n_iter: int = Field(
        default=100,
        ge=50,
        le=500,
        description="HMM training iterations",
    )

    # =========================================================================
    # Volatility Parameters
    # =========================================================================
    vol_window: int = Field(
        default=20,
        ge=5,
        le=60,
        description="Rolling volatility window",
    )
    vol_target: float = Field(
        default=0.40,
        ge=0.05,
        le=1.0,
        description="Annual volatility target",
    )
    min_volatility: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="Minimum volatility clamp (prevent division by zero)",
    )
    annualization_factor: float = Field(
        default=365.0,
        gt=0,
        description="Annualization factor (daily: 365, 4h: 2190, 1h: 8760)",
    )

    # =========================================================================
    # Options
    # =========================================================================
    use_log_returns: bool = Field(
        default=True,
        description="Whether to use log returns (recommended: True)",
    )
    atr_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ATR calculation period (for Trailing Stop)",
    )

    # =========================================================================
    # Short Mode Settings
    # =========================================================================
    short_mode: ShortMode = Field(
        default=ShortMode.DISABLED,
        description="Short position handling mode (DISABLED/HEDGE_ONLY/FULL)",
    )
    hedge_threshold: float = Field(
        default=-0.07,
        ge=-0.30,
        le=-0.05,
        description="Hedge short activation drawdown threshold (e.g., -0.07 = -7%)",
    )
    hedge_strength_ratio: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="Hedge short strength ratio (vs long, e.g., 0.8 = 80%)",
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """Configuration consistency validation.

        Returns:
            Validated self

        Raises:
            ValueError: If configuration is inconsistent
        """
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.min_train_window < self.vol_window:
            msg = f"min_train_window ({self.min_train_window}) must be >= vol_window ({self.vol_window})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """Required warmup period (number of candles).

        Returns:
            Minimum number of candles needed before strategy calculation
        """
        return self.min_train_window + 1
