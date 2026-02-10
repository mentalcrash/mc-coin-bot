"""Acceleration-Skewness Signal 전략 설정.

가격 가속도가 양(+)이고 rolling skewness도 양(+)이면 우상향 테일이 reward로 전환.
Skewness가 음(-)이면 crash risk -> 거래 중단.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class AccelSkewConfig(BaseModel):
    """Acceleration-Skewness Signal 전략 설정.

    Attributes:
        acc_smooth_window: Acceleration smoothing window (bars).
        skew_window: Rolling skewness window (bars).
        skew_threshold: Skewness 진입 임계값.
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 12H TF 연환산 계수 (730).
        atr_period: ATR 계산 기간.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    acc_smooth_window: int = Field(default=12, ge=3, le=60)
    skew_window: int = Field(default=30, ge=10, le=120)
    skew_threshold: float = Field(default=0.3, ge=0.0, le=2.0)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=730.0, gt=0.0)

    # --- Options ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> AccelSkewConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.acc_smooth_window, self.skew_window, self.vol_window, self.atr_period) + 10
