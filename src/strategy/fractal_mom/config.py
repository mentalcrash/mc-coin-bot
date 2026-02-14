"""Fractal-Filtered Momentum 전략 설정.

Fractal Market Hypothesis(Peters 1994). D<1.5 = deterministic regime에서만
trend following 활성화. fractal_dimension을 레짐 오버레이로 사용.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class FractalMomConfig(BaseModel):
    """Fractal-Filtered Momentum 전략 설정.

    Attributes:
        fractal_period: Fractal dimension 계산 기간.
        fractal_threshold: D < threshold → deterministic (trend).
        mom_fast: 빠른 모멘텀 lookback.
        mom_slow: 느린 모멘텀 lookback.
        er_period: Efficiency ratio 계산 기간 (추가 확인).
        er_threshold: ER > threshold → trend 확인.
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 4H = 2190.0.
        atr_period: ATR 계산 기간.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY drawdown 기준.
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄.
    """

    model_config = ConfigDict(frozen=True)

    # --- Fractal Parameters ---
    fractal_period: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Fractal dimension calculation period.",
    )
    fractal_threshold: float = Field(
        default=1.5,
        ge=1.1,
        le=1.9,
        description="D < threshold = deterministic regime.",
    )

    # --- Momentum Parameters ---
    mom_fast: int = Field(
        default=12,
        ge=3,
        le=50,
        description="Fast momentum lookback (bars).",
    )
    mom_slow: int = Field(
        default=48,
        ge=10,
        le=200,
        description="Slow momentum lookback (bars).",
    )

    # --- Efficiency Ratio (confirming) ---
    er_period: int = Field(
        default=10,
        ge=3,
        le=50,
        description="Efficiency ratio calculation period.",
    )
    er_threshold: float = Field(
        default=0.3,
        ge=0.1,
        le=0.8,
        description="ER > threshold = trend confirmed.",
    )

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=2190.0, gt=0)

    # --- Options ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> Self:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.mom_fast >= self.mom_slow:
            msg = f"mom_fast ({self.mom_fast}) must be < mom_slow ({self.mom_slow})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.fractal_period, self.mom_slow, self.vol_window, self.atr_period) + 10
