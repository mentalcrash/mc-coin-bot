"""Cascade Momentum 전략 설정.

연속 동방향 4H 리턴(streak)은 허딩/FOMO 캐스케이드를 반영하며 방향 지속.
스트릭 카운트 x 평균 body / ATR 정규화로 시그널 생성.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class CascadeMomConfig(BaseModel):
    """Cascade Momentum 전략 설정.

    Attributes:
        min_streak: 최소 연속 동방향 bar 수.
        cascade_window: Cascade score rolling window.
        atr_period: ATR 정규화 기간.
        score_threshold: 롱/숏 진입 cascade score 임계값.
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 4H TF 연환산 계수 (2190).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    min_streak: int = Field(default=3, ge=2, le=10)
    cascade_window: int = Field(default=10, ge=3, le=50)
    atr_period: int = Field(default=14, ge=5, le=50)
    score_threshold: float = Field(default=1.5, ge=0.1, le=5.0)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=2190.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> CascadeMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.cascade_window, self.atr_period, self.vol_window) + 10
