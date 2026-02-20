"""Realized-Parkinson Vol Regime 전략 설정.

Realized Vol과 Parkinson Vol의 비율로 시장 미시구조 상태를 측정.
PV/RV 높음 = 장중 변동 크나 종가 변화 작음 = 축적/분배 단계.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class RpVolRegimeConfig(BaseModel):
    """Realized-Parkinson Vol Regime 전략 설정.

    Attributes:
        rv_window: Realized Vol 계산 기간.
        pv_window: Parkinson Vol 계산 기간.
        ratio_zscore_window: PV/RV 비율 z-score 계산 기간.
        momentum_window: 모멘텀 확인 기간.
        ratio_upper: PV/RV 높음 임계 z-score (축적/분배).
        ratio_lower: PV/RV 낮음 임계 z-score (추세 지속).
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
    rv_window: int = Field(default=20, ge=5, le=100)
    pv_window: int = Field(default=20, ge=5, le=100)
    ratio_zscore_window: int = Field(default=60, ge=20, le=252)
    momentum_window: int = Field(default=21, ge=5, le=60)
    ratio_upper: float = Field(default=1.0, ge=0.0, le=3.0)
    ratio_lower: float = Field(default=-1.0, ge=-3.0, le=0.0)

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
    def _validate_cross_fields(self) -> RpVolRegimeConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.rv_window, self.pv_window, self.ratio_zscore_window, self.vol_window) + 10
