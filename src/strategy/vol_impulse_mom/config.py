"""Volume-Impulse Momentum 전략 설정.

비정상 거래량(3x+ spike)이 방향성 가격 이동과 동시 발생 시
informed trading이며 continuation 유발. Vol-Climax(반전 가설)의 정반대 접근.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class VolImpulseMomConfig(BaseModel):
    """Volume-Impulse Momentum 전략 설정.

    Attributes:
        vol_spike_window: Volume spike 판별 rolling window.
        vol_spike_multiplier: Volume spike 배수 (평균 대비).
        body_ratio_threshold: 방향성 bar 확인 body ratio 임계값.
        hold_bars: Impulse 이후 포지션 유지 bar 수.
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 15m TF 연환산 계수 (35040).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    vol_spike_window: int = Field(default=20, ge=5, le=100)
    vol_spike_multiplier: float = Field(default=3.0, ge=1.5, le=10.0)
    body_ratio_threshold: float = Field(default=0.5, ge=0.1, le=0.9)
    hold_bars: int = Field(default=4, ge=1, le=20)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=35040.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> VolImpulseMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.vol_spike_window, self.vol_window) + self.hold_bars + 10
