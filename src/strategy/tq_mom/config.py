"""Trend Quality Momentum (TQ-Mom) 전략 설정.

Hurst exponent(지속성)과 Fractal Dimension(복잡도) 조합으로 추세 품질 판별.
높은 품질 추세(H > threshold, FD < threshold)에서만 momentum 진입.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class TqMomConfig(BaseModel):
    """Trend Quality Momentum (TQ-Mom) 전략 설정.

    Signal Logic:
        1. Hurst exponent > hurst_threshold → 추세 지속성 확인
        2. Fractal Dimension < fd_threshold → 낮은 복잡도 = 정돈된 추세
        3. quality_pass = (hurst > hurst_threshold) & (fd < fd_threshold)
        4. direction = sign(price_mom) where quality_pass, else 0
        5. conviction = (hurst - 0.5).clip(0, 0.5) * (2 - fd).clip(0, 1)
           → H가 0.5 초과할수록, FD가 1에 가까울수록 높은 확신
        6. strength = direction * vol_scalar * conviction

    Attributes:
        hurst_window: Hurst exponent rolling window (bars).
        hurst_threshold: 최소 Hurst 값 (> 0.5 = 추세 지속).
        fd_period: Fractal Dimension 계산 period.
        fd_threshold: 최대 FD 값 (< 1.5 = 정돈된 움직임).
        mom_lookback: 가격 모멘텀 lookback (bars).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 12H TF 연환산 계수 (730).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    hurst_window: int = Field(default=40, ge=20, le=200)
    hurst_threshold: float = Field(default=0.55, ge=0.5, le=0.9)
    fd_period: int = Field(default=20, ge=10, le=100)
    fd_threshold: float = Field(default=1.4, ge=1.0, le=1.9)
    mom_lookback: int = Field(default=20, ge=3, le=100)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=730.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> TqMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        # fractal_dimension은 내부적으로 2 * fd_period window 사용
        return (
            max(
                self.hurst_window,
                2 * self.fd_period,
                self.mom_lookback,
                self.vol_window,
            )
            + 10
        )
