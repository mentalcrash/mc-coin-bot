"""Return Distribution Momentum 전략 설정.

수익률 분포에서 양수 비율(positive day proportion)으로 모멘텀 지속성 예측.
행동 편향 - magnitude 과잉반응, breadth 과소반응 활용.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class DistMomConfig(BaseModel):
    """Return Distribution Momentum 전략 설정.

    Attributes:
        dist_window: 수익률 분포 측정 rolling window.
        long_threshold: long 진입 양수비율 임계값.
        short_threshold: short 진입 양수비율 임계값 (하한).
        momentum_window: 모멘텀 확인 기간 (ROC).
        skew_window: 수익률 skewness 계산 기간.
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
    dist_window: int = Field(default=21, ge=5, le=100)
    long_threshold: float = Field(default=0.60, ge=0.50, le=0.90)
    short_threshold: float = Field(default=0.40, ge=0.10, le=0.50)
    momentum_window: int = Field(default=21, ge=5, le=60)
    skew_window: int = Field(default=21, ge=5, le=100)

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
    def _validate_cross_fields(self) -> DistMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.dist_window, self.momentum_window, self.skew_window, self.vol_window) + 10
