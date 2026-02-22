"""Drawdown-Recovery Phase 전략 설정.

드로다운 후 50%+ 회복 시, 손실회피 투매자의 과소반응으로 follow-through 모멘텀 발생.
Prospect theory 기반.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class DDRecoveryPhaseConfig(BaseModel):
    """Drawdown-Recovery Phase 전략 설정.

    Attributes:
        dd_threshold: Drawdown 진입 기준 (음수, e.g. -0.15 = -15%).
        recovery_ratio: Drawdown 대비 회복 비율 기준 (0~1, e.g. 0.5 = 50% 회복).
        dd_lookback: Drawdown 계산 rolling max lookback (bars).
        momentum_lookback: 회복 후 momentum confirmation lookback.
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 1D TF 연환산 계수 (365).
        atr_period: ATR 계산 기간.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    dd_threshold: float = Field(default=-0.15, le=-0.05, ge=-0.50)
    recovery_ratio: float = Field(default=0.50, ge=0.2, le=0.9)
    dd_lookback: int = Field(default=60, ge=20, le=200)
    momentum_lookback: int = Field(default=10, ge=3, le=30)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=365.0, gt=0.0)

    # --- Options ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> DDRecoveryPhaseConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.dd_lookback, self.vol_window, self.atr_period, self.momentum_lookback) + 10
