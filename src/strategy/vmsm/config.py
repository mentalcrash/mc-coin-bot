"""Volume-Gated Multi-Scale Momentum 전략 설정.

다중 시간 수평(1D/3D/1W) ROC 앙상블 + 볼륨 게이트로 모멘텀 conviction 강화.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class VmsmConfig(BaseModel):
    """Volume-Gated Multi-Scale Momentum 전략 설정.

    Attributes:
        roc_short: 단기 ROC bars (3H 기준 ~1D).
        roc_mid: 중기 ROC bars (3H 기준 ~3D).
        roc_long: 장기 ROC bars (3H 기준 ~1W).
        vol_gate_window: Volume gate rolling window.
        vol_gate_multiplier: Volume spike multiplier (above avg).
        ensemble_threshold: 앙상블 시그널 임계값 (합의 수).
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 3H TF 연환산 계수 (2920).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    roc_short: int = Field(default=8, ge=2, le=30)
    roc_mid: int = Field(default=24, ge=5, le=80)
    roc_long: int = Field(default=56, ge=10, le=200)
    vol_gate_window: int = Field(default=20, ge=5, le=100)
    vol_gate_multiplier: float = Field(default=1.2, ge=0.5, le=5.0)
    ensemble_threshold: int = Field(default=2, ge=1, le=3)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=2920.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> VmsmConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if not (self.roc_short < self.roc_mid < self.roc_long):
            msg = f"ROC periods must be ordered: short({self.roc_short}) < mid({self.roc_mid}) < long({self.roc_long})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.roc_long, self.vol_gate_window, self.vol_window) + 10
