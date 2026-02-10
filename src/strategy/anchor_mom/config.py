"""Anchored Momentum 전략 설정.

Rolling high 대비 근접도(nearness)와 모멘텀 방향을 결합하여
심리적 앵커링 효과를 포착한다.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class AnchorMomConfig(BaseModel):
    """Anchored Momentum 전략 설정.

    Attributes:
        nearness_lookback: Rolling max lookback (bars).
        mom_lookback: Momentum direction lookback (bars).
        strong_nearness: Strong long nearness 임계값 (0~1).
        weak_nearness: Weak long nearness 임계값 (0~1).
        short_nearness: Short 진입 nearness 임계값 (0~1).
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
    nearness_lookback: int = Field(default=60, ge=10, le=200)
    mom_lookback: int = Field(default=30, ge=5, le=120)
    strong_nearness: float = Field(default=0.95, ge=0.8, le=1.0)
    weak_nearness: float = Field(default=0.85, ge=0.7, le=0.95)
    short_nearness: float = Field(default=0.80, ge=0.5, le=0.9)

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
    def _validate_cross_fields(self) -> AnchorMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.strong_nearness <= self.weak_nearness:
            msg = f"strong_nearness ({self.strong_nearness}) must be > weak_nearness ({self.weak_nearness})"
            raise ValueError(msg)
        if self.weak_nearness <= self.short_nearness:
            msg = f"weak_nearness ({self.weak_nearness}) must be > short_nearness ({self.short_nearness})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.nearness_lookback, self.mom_lookback, self.vol_window, self.atr_period) + 10
