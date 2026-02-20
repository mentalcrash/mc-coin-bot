"""Complexity-Filtered Trend 전략 설정.

Fractal dimension + Hurst exponent + efficiency ratio로 시장 복잡도/예측가능성 정량화.
CTREND과 직교하는 정보이론 기반. 학술근거: MDPI Entropy 2025.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class ComplexityTrendConfig(BaseModel):
    """Complexity-Filtered Trend 전략 설정.

    Fractal dimension, Hurst exponent, Efficiency ratio 세 가지
    정보이론 지표로 시장 복잡도를 정량화하고, 낮은 복잡도(= 높은 예측가능성)
    구간에서만 추세추종 시그널을 활성화합니다.

    Attributes:
        hurst_window: Hurst exponent rolling window.
        fractal_period: Fractal dimension 계산 기간.
        er_period: Efficiency ratio 계산 기간.
        trend_window: 추세 확인용 SMA 기간.
        hurst_threshold: Hurst > threshold → 추세 지속 환경 (0.5 기준).
        fractal_threshold: Fractal < threshold → 낮은 복잡도 (1.5 미만).
        er_threshold: ER > threshold → 효율적 추세 (0~1).
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
    hurst_window: int = Field(default=60, ge=20, le=252)
    fractal_period: int = Field(default=30, ge=10, le=120)
    er_period: int = Field(default=21, ge=5, le=60)
    trend_window: int = Field(default=42, ge=10, le=120)
    hurst_threshold: float = Field(default=0.55, ge=0.40, le=0.80)
    fractal_threshold: float = Field(default=1.45, ge=1.10, le=1.80)
    er_threshold: float = Field(default=0.15, ge=0.05, le=0.50)

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
    def _validate_cross_fields(self) -> ComplexityTrendConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        # fractal_dimension uses 2 * fractal_period internally
        return (
            max(
                self.hurst_window,
                2 * self.fractal_period,
                self.er_period,
                self.trend_window,
                self.vol_window,
            )
            + 10
        )
