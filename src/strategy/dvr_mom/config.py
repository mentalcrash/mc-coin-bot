"""Vol-Efficiency Momentum 전략 설정.

Parkinson(High-Low) vol과 Close-to-Close vol 비율로 바 내부 효율성을 측정.
비율 낮으면 방향성 바(추세 확인), 높으면 혼조 바(리스크 축소).
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class DvrMomConfig(BaseModel):
    """Vol-Efficiency Momentum 전략 설정.

    Attributes:
        dvr_window: Directional Vol Ratio 계산 rolling window.
        mom_lookback: 모멘텀 방향 lookback (bars).
        dvr_threshold: DVR 임계값 (이하면 방향성 바).
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
    dvr_window: int = Field(default=20, ge=5, le=100)
    mom_lookback: int = Field(default=20, ge=5, le=120)
    dvr_threshold: float = Field(default=0.8, gt=0.0, le=2.0)

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
    def _validate_cross_fields(self) -> DvrMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.dvr_window, self.mom_lookback, self.vol_window) + 10
