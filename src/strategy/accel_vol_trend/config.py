"""Acceleration-Volatility Trend 전략 설정.

2차 미분(가속도)으로 모멘텀 품질 캡처 + GK vol 정규화.
학술근거: FRL 2025 risk-managed momentum Sharpe 27% 향상.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class AccelVolTrendConfig(BaseModel):
    """Acceleration-Volatility Trend 전략 설정.

    가격 가속도(2차 미분)와 GK volatility 정규화를 결합하여
    모멘텀 품질을 측정합니다. 가속 + 저변동성 = 고품질 모멘텀.

    Attributes:
        accel_fast: 가속도 계산 fast ROC 기간.
        accel_slow: 가속도 계산 slow ROC 기간.
        gk_window: GK vol rolling window.
        accel_smooth: 가속도 smoothing window.
        accel_long_threshold: 가속도 long 임계값 (양수: 가속 중).
        accel_short_threshold: 가속도 short 임계값 (음수: 감속 중).
        momentum_window: 추세 확인용 ROC 기간.
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
    accel_fast: int = Field(default=5, ge=2, le=21)
    accel_slow: int = Field(default=21, ge=10, le=60)
    gk_window: int = Field(default=21, ge=5, le=100)
    accel_smooth: int = Field(default=10, ge=3, le=30)
    accel_long_threshold: float = Field(default=0.005, ge=0.0, le=0.10)
    accel_short_threshold: float = Field(default=-0.005, ge=-0.10, le=0.0)
    momentum_window: int = Field(default=21, ge=5, le=60)

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
    def _validate_cross_fields(self) -> AccelVolTrendConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.accel_fast >= self.accel_slow:
            msg = f"accel_fast ({self.accel_fast}) must be < accel_slow ({self.accel_slow})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            max(
                self.accel_slow,
                self.gk_window,
                self.momentum_window,
                self.vol_window,
            )
            + self.accel_smooth
            + 10
        )
