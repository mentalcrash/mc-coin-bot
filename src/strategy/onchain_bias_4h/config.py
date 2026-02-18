"""On-chain Bias 4H 전략 설정.

On-chain Phase(MVRV/Flow/Stablecoin) 1D gate + 4H momentum timing.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class OnchainBias4hConfig(BaseModel):
    """On-chain Bias 4H 전략 설정.

    Attributes:
        mvrv_accumulation: MVRV 축적 임계값 (이하 = accumulation).
        mvrv_distribution: MVRV 분배 임계값 (이상 = distribution).
        stab_roc_window: Stablecoin ROC window (4H bars).
        er_window: Efficiency Ratio window.
        er_min: ER 최소 (모멘텀 확인).
        roc_window: Price ROC window (4H bars).
        roc_threshold: Price ROC 임계값.
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 4H TF 연환산 계수.
        atr_period: ATR 계산 기간.
        short_mode: 숏 포지션 허용 모드.
    """

    model_config = ConfigDict(frozen=True)

    # --- Phase Thresholds ---
    mvrv_accumulation: float = Field(default=1.8, ge=0.5, le=2.5)
    mvrv_distribution: float = Field(default=3.0, ge=2.0, le=5.0)
    stab_roc_window: int = Field(default=42, ge=12, le=120)

    # --- Momentum Timing ---
    er_window: int = Field(default=30, ge=5, le=120)
    er_min: float = Field(default=0.3, ge=0.1, le=0.6)
    roc_window: int = Field(default=15, ge=5, le=30)
    roc_threshold: float = Field(default=0.02, ge=0.005, le=0.10)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=120)
    min_volatility: float = Field(default=0.05, gt=0.0, le=0.20)
    annualization_factor: float = Field(default=2190.0, gt=0.0)

    # --- ATR ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> OnchainBias4hConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.mvrv_accumulation >= self.mvrv_distribution:
            msg = (
                f"mvrv_accumulation ({self.mvrv_accumulation}) "
                f"must be < mvrv_distribution ({self.mvrv_distribution})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.stab_roc_window, self.er_window, self.vol_window, self.roc_window) + 10
