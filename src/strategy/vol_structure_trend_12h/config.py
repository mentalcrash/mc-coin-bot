"""Vol-Structure-Trend 12H 전략 설정.

3종 변동성 추정기(GK/PK/YZ) 합의 기반 추세 감지.
변동성 측정 직교성으로 노이즈 필터링 — multi-scale 앙상블.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class VolStructureTrendConfig(BaseModel):
    """Vol-Structure-Trend 12H 전략 설정.

    Attributes:
        scale_short: 단기 변동성 측정 lookback (bars).
        scale_mid: 중기 변동성 측정 lookback (bars).
        scale_long: 장기 변동성 측정 lookback (bars).
        roc_lookback: ROC 모멘텀 방향 lookback (bars).
        vol_agreement_threshold: 3종 추정기 합의 임계값 (0~1).
        vol_expansion_percentile: 변동성 확대 퍼센타일 임계값.
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 12H TF 연환산 계수 (730).
        atr_period: ATR 계산 기간.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Multi-Scale Vol Parameters ---
    scale_short: int = Field(default=14, ge=5, le=50)
    scale_mid: int = Field(default=30, ge=10, le=100)
    scale_long: int = Field(default=60, ge=20, le=200)

    # --- Momentum Direction ---
    roc_lookback: int = Field(default=20, ge=5, le=100)

    # --- Vol Agreement ---
    vol_agreement_threshold: float = Field(default=0.6, ge=0.3, le=1.0)
    vol_expansion_percentile: float = Field(default=60.0, ge=30.0, le=90.0)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=730.0, gt=0.0)

    # --- Options ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> VolStructureTrendConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if not (self.scale_short < self.scale_mid < self.scale_long):
            msg = f"scale_short < scale_mid < scale_long required: {self.scale_short} < {self.scale_mid} < {self.scale_long}"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.scale_long, self.vol_window, self.roc_lookback) + 10
