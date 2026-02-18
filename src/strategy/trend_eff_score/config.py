"""Trend Efficiency Scorer 전략 설정.

ER(Efficiency Ratio)로 추세 품질을 측정하고,
품질 높을 때만 다중 수평선 모멘텀(ROC) 합의로 방향 결정.
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


class TrendEffScoreConfig(BaseModel):
    """Trend Efficiency Scorer 전략 설정.

    Attributes:
        er_window: Efficiency Ratio 계산 기간 (bars).
        roc_short: 단기 ROC 기간 (~3일 at 4H).
        roc_medium: 중기 ROC 기간 (~7일 at 4H).
        roc_long: 장기 ROC 기간 (~15일 at 4H).
        er_threshold: ER 활성화 임계값 (이 이상이면 trending).
        min_score: 최소 다수결 동의 수 (|score| >= min_score → 방향 확정).
        adx_period: ADX 기간 (추세 확인용).
        adx_threshold: ADX 활성화 임계값.
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 4H TF 연환산 계수.
        atr_period: ATR 계산 기간.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY drawdown 기준.
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    er_window: int = Field(default=30, ge=10, le=120)
    roc_short: int = Field(default=18, ge=6, le=42)
    roc_medium: int = Field(default=42, ge=18, le=90)
    roc_long: int = Field(default=90, ge=42, le=180)
    er_threshold: float = Field(default=0.25, ge=0.1, le=0.5)
    min_score: int = Field(default=2, ge=1, le=3)
    adx_period: int = Field(default=14, ge=5, le=30)
    adx_threshold: float = Field(default=20.0, ge=5.0, le=50.0)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=2190.0, gt=0.0)

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
        if not (self.roc_short < self.roc_medium < self.roc_long):
            msg = (
                f"ROC periods must be strictly increasing: "
                f"roc_short ({self.roc_short}) < roc_medium ({self.roc_medium}) < roc_long ({self.roc_long})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.er_window, self.roc_long, self.vol_window, self.adx_period) + 10
