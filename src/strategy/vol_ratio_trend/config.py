"""Volatility Ratio Trend 전략 설정.

단기/장기 RV 비율(vol term structure)로 시장 스트레스 상태 측정.
Contango(단기<장기) = 모멘텀 신뢰, Backwardation = 스트레스.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class VolRatioTrendConfig(BaseModel):
    """Volatility Ratio Trend 전략 설정.

    Attributes:
        short_vol_window: 단기 변동성 기간.
        long_vol_window: 장기 변동성 기간.
        ratio_smooth_window: vol ratio smoothing window.
        contango_threshold: contango 판단 임계값 (ratio < threshold).
        backwardation_threshold: backwardation 판단 임계값 (ratio > threshold).
        momentum_window: 모멘텀 확인 기간 (ROC).
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
    short_vol_window: int = Field(default=10, ge=3, le=30)
    long_vol_window: int = Field(default=60, ge=20, le=200)
    ratio_smooth_window: int = Field(default=5, ge=3, le=20)
    contango_threshold: float = Field(default=0.90, ge=0.50, le=1.0)
    backwardation_threshold: float = Field(default=1.20, ge=1.0, le=2.0)
    momentum_window: int = Field(default=21, ge=5, le=60)

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
    def _validate_cross_fields(self) -> VolRatioTrendConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.short_vol_window >= self.long_vol_window:
            msg = (
                f"short_vol_window ({self.short_vol_window}) "
                f"must be < long_vol_window ({self.long_vol_window})"
            )
            raise ValueError(msg)
        if self.contango_threshold >= self.backwardation_threshold:
            msg = (
                f"contango_threshold ({self.contango_threshold}) "
                f"must be < backwardation_threshold ({self.backwardation_threshold})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            max(
                self.long_vol_window + self.ratio_smooth_window,
                self.momentum_window,
                self.vol_window,
            )
            + 10
        )
