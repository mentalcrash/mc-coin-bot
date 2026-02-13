"""Directional Volume Trend 전략 설정.

Up-bar/down-bar 거래량 비율로 방향적 conviction 측정 기반 추세추종.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class DirVolTrendConfig(BaseModel):
    """Directional Volume Trend 전략 설정.

    Attributes:
        dvt_window: Up/down volume ratio rolling window.
        dvt_smooth: Volume ratio EMA smoothing span.
        dvt_long_threshold: 롱 진입 volume ratio 임계값.
        dvt_short_threshold: 숏 진입 volume ratio 임계값.
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 6H TF 연환산 계수 (1460).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    dvt_window: int = Field(default=20, ge=5, le=100)
    dvt_smooth: int = Field(default=5, ge=1, le=30)
    dvt_long_threshold: float = Field(default=1.3, ge=1.0, le=5.0)
    dvt_short_threshold: float = Field(default=0.7, gt=0.0, le=1.0)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=1460.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> DirVolTrendConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.dvt_window, self.vol_window) + self.dvt_smooth + 10
