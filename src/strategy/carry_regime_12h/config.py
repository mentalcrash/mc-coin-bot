"""Carry-Regime Trend 전략 설정.

12H multi-scale EMA trend entry + FR 90일 percentile로 exit speed 적응 조절.
BIS WP 1087: extreme carry = crash risk. carry_sensitivity=0이면 pure trend baseline.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class CarryRegimeConfig(BaseModel):
    """Carry-Regime Trend 전략 설정.

    Attributes:
        ema_fast: Fast EMA 기간 (bars).
        ema_mid: Medium EMA 기간 (bars).
        ema_slow: Slow EMA 기간 (bars).
        fr_percentile_window: FR percentile 계산 rolling window (bars).
        carry_sensitivity: FR percentile의 exit 영향 강도 (0=pure trend).
        exit_base_threshold: 기본 exit threshold (EMA alignment score).
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

    # --- Multi-Scale EMA Parameters ---
    ema_fast: int = Field(default=8, ge=3, le=30)
    ema_mid: int = Field(default=21, ge=10, le=60)
    ema_slow: int = Field(default=55, ge=30, le=150)

    # --- FR Percentile Parameters ---
    fr_percentile_window: int = Field(default=135, ge=30, le=400)
    carry_sensitivity: float = Field(default=0.5, ge=0.0, le=2.0)
    exit_base_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=730.0, gt=0.0)

    # --- Options ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> CarryRegimeConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.ema_fast >= self.ema_mid:
            msg = f"ema_fast ({self.ema_fast}) must be < ema_mid ({self.ema_mid})"
            raise ValueError(msg)
        if self.ema_mid >= self.ema_slow:
            msg = f"ema_mid ({self.ema_mid}) must be < ema_slow ({self.ema_slow})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.ema_slow, self.fr_percentile_window, self.vol_window) + 10
