"""F&G Persistence Break 전략 설정.

F&G 극단 구간의 persistence break(탈출 시점)이 방향 전환 시그널.
기존 레벨 기반과 차별화: 극단에 N일 체류 후 탈출하는 시점을 포착.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class FgPersistBreakConfig(BaseModel):
    """F&G Persistence Break 전략 설정.

    Attributes:
        fear_threshold: Fear zone 진입 임계값 (F&G < threshold).
        greed_threshold: Greed zone 진입 임계값 (F&G > threshold).
        min_persist: 최소 극단 구간 체류 기간 (break 시그널 발생 조건).
        max_streak_cap: Streak 길이 정규화 상한 (strength 계산용).
        price_mom_window: 가격 모멘텀 확인 윈도우.
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
    fear_threshold: float = Field(default=25.0, ge=5.0, le=50.0)
    greed_threshold: float = Field(default=75.0, ge=50.0, le=95.0)
    min_persist: int = Field(default=5, ge=2, le=30)
    max_streak_cap: int = Field(default=20, ge=5, le=60)
    price_mom_window: int = Field(default=5, ge=2, le=20)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=365.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> FgPersistBreakConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.fear_threshold >= self.greed_threshold:
            msg = f"fear_threshold ({self.fear_threshold}) must be < greed_threshold ({self.greed_threshold})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.min_persist + self.price_mom_window, self.vol_window) + 10
