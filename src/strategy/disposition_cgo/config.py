"""Disposition CGO 전략 설정.

Grinblatt-Han(2005) turnover-weighted reference price 기반 Capital Gains Overhang와
Frazzini(2006) overhang spread를 결합하여 disposition effect underreaction drift 포착.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class DispositionCgoConfig(BaseModel):
    """Disposition CGO 전략 설정.

    Attributes:
        turnover_window: Reference price EWM 가중 window (bars).
        cgo_smooth_window: CGO smoothing EMA window.
        momentum_window: Momentum confirmation window.
        overhang_spread_window: Overhang spread rolling window.
        cgo_entry_threshold: CGO 절대값 진입 임계값.
        spread_confirm_threshold: Overhang spread confirmation threshold.
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
    turnover_window: int = Field(default=60, ge=10, le=365)
    cgo_smooth_window: int = Field(default=10, ge=3, le=100)
    momentum_window: int = Field(default=20, ge=5, le=200)
    overhang_spread_window: int = Field(default=60, ge=10, le=365)
    cgo_entry_threshold: float = Field(default=0.03, ge=0.0, le=1.0)
    spread_confirm_threshold: float = Field(default=0.0, ge=-1.0, le=1.0)

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
    def _validate_cross_fields(self) -> DispositionCgoConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            max(
                self.turnover_window,
                self.overhang_spread_window,
                self.vol_window,
                self.momentum_window,
            )
            + self.cgo_smooth_window
            + 10
        )
