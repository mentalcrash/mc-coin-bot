"""Autocorrelation Momentum 전략 설정.

lag-1 자기상관이 양이면 모멘텀 레짐(정보 천천히 반영 중) -> 추세추종 유효.
Lo(2004) Adaptive Market Hypothesis 기반.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class AutocorrMomConfig(BaseModel):
    """Autocorrelation Momentum 전략 설정.

    Attributes:
        autocorr_window: 자기상관 계산 rolling window.
        momentum_window: 모멘텀(ROC) 계산 기간.
        autocorr_threshold: 자기상관 양수 확인 임계값 (>0 = 모멘텀 레짐).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: TF별 연환산 계수 (1D=365).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    autocorr_window: int = Field(default=20, ge=5, le=100)
    momentum_window: int = Field(default=21, ge=5, le=100)
    autocorr_threshold: float = Field(default=0.0, ge=-0.5, le=0.5)

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
    def _validate_cross_fields(self) -> AutocorrMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.autocorr_window, self.momentum_window, self.vol_window) + 10
