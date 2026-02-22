"""Return Streak Persistence 전략 설정.

연속 양봉 3일+ → 군집 효과(FOMO)로 추가 상승. 연속 음봉 3일+ → 패닉 확산.
Crypto momentum 효과 실증 기반.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class StreakPersistenceConfig(BaseModel):
    """Return Streak Persistence 전략 설정.

    Attributes:
        streak_threshold: 시그널 발동 최소 연속 bar 수.
        max_streak_cap: Streak conviction 상한 bar 수 (과적합 방지).
        momentum_lookback: 모멘텀 방향 확인 lookback.
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 1D TF 연환산 계수 (365).
        atr_period: ATR 계산 기간.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    streak_threshold: int = Field(default=3, ge=2, le=10)
    max_streak_cap: int = Field(default=7, ge=3, le=15)
    momentum_lookback: int = Field(default=20, ge=5, le=60)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=365.0, gt=0.0)

    # --- Options ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> StreakPersistenceConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.max_streak_cap < self.streak_threshold:
            msg = (
                f"max_streak_cap ({self.max_streak_cap}) must be >= "
                f"streak_threshold ({self.streak_threshold})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.momentum_lookback, self.vol_window, self.atr_period) + 10
