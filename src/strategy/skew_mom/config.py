"""Skew-Gated Momentum 전략 설정.

수익률 분포 비대칭성(skewness)으로 모멘텀 방향 지속 가능성을 예측.
양의 skew + 상승 모멘텀 = 상방 tail 가능성 확인, 음의 skew 전환 = 크래시 전조.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class SkewMomConfig(BaseModel):
    """Skew-Gated Momentum 전략 설정.

    Attributes:
        skew_window: Rolling skewness 계산 윈도우.
        mom_lookback: 모멘텀 방향 lookback (bars).
        skew_long_threshold: 롱 진입 skewness 임계값.
        skew_short_threshold: 숏 진입 skewness 임계값 (음수).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 1D TF 연환산 계수 (365).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    skew_window: int = Field(default=30, ge=10, le=120)
    mom_lookback: int = Field(default=20, ge=5, le=120)
    skew_long_threshold: float = Field(default=0.3, ge=0.0, le=3.0)
    skew_short_threshold: float = Field(default=-0.3, ge=-3.0, le=0.0)

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
    def _validate_cross_fields(self) -> SkewMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.skew_window, self.mom_lookback, self.vol_window) + 10
