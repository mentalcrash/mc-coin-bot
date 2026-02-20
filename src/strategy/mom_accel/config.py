"""Momentum Acceleration 전략 설정.

모멘텀의 2차 미분(가속도)으로 추세 성숙도 측정.
속도와 가속도가 alignment될 때만 추종.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class MomAccelConfig(BaseModel):
    """Momentum Acceleration 전략 설정.

    Attributes:
        fast_roc: 빠른 ROC 기간 (velocity).
        slow_roc: 느린 ROC 기간 (velocity).
        accel_window: 가속도 smoothing window.
        momentum_window: 모멘텀(속도) 확인 기간.
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 1D TF 연환산 계수.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준.
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    fast_roc: int = Field(default=10, ge=3, le=30)
    slow_roc: int = Field(default=30, ge=10, le=60)
    accel_window: int = Field(default=5, ge=3, le=20)
    momentum_window: int = Field(default=21, ge=5, le=60)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=365.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> MomAccelConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.fast_roc >= self.slow_roc:
            msg = f"fast_roc ({self.fast_roc}) must be < slow_roc ({self.slow_roc})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.slow_roc + self.accel_window, self.momentum_window, self.vol_window) + 10
