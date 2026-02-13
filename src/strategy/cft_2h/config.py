"""Conviction-Filtered Trend 전략 설정.

UP->UP 레짐 전환을 모멘텀 진입 게이트로 사용 -- 레짐을 필터(신호 아님)로 활용.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class Cft2hConfig(BaseModel):
    """Conviction-Filtered Trend 전략 설정.

    Attributes:
        regime_window: 레짐 판별 rolling window.
        mom_lookback: 모멘텀 lookback.
        regime_up_threshold: 양수 수익률 비율 레짐 기준.
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 2H TF 연환산 계수 (4380).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    regime_window: int = Field(default=24, ge=5, le=120)
    mom_lookback: int = Field(default=12, ge=3, le=60)
    regime_up_threshold: float = Field(default=0.55, ge=0.5, le=0.8)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=4380.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> Cft2hConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.regime_window, self.mom_lookback, self.vol_window) + 10
