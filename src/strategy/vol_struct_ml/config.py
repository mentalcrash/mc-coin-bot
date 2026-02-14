"""Volatility Structure ML 전략 설정.

Vol clustering(Mandelbrot 1963) + VoV premium(Bollerslev 2009).
GK/Parkinson/YZ vol, VoV, fractal_dimension, hurst, efficiency_ratio, ADX 등
13종 vol 기반 feature space로 4H 방향성 예측.
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


class VolStructMLConfig(BaseModel):
    """Volatility Structure ML 전략 설정.

    Attributes:
        training_window: Rolling Elastic Net 학습 윈도우 (bar 수).
        prediction_horizon: Forward return 예측 기간 (bar 수).
        alpha: Elastic Net L1 ratio (0=Ridge, 1=Lasso).
        vol_estimator_window: GK/Parkinson/YZ 변동성 추정 윈도우.
        fractal_period: Fractal dimension 계산 기간.
        hurst_window: Hurst exponent 계산 윈도우.
        er_period: Efficiency ratio 계산 기간.
        adx_period: ADX 계산 기간.
        vov_window: Volatility-of-volatility 계산 윈도우.
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 4H = 2190.0.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준.
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- ML Parameters ---
    training_window: int = Field(
        default=252,
        ge=60,
        le=504,
        description="Rolling Elastic Net training window (bars).",
    )
    prediction_horizon: int = Field(
        default=6,
        ge=1,
        le=30,
        description="Forward return prediction horizon (bars).",
    )
    alpha: float = Field(
        default=0.5,
        ge=0.01,
        le=1.0,
        description="Elastic Net L1 ratio.",
    )

    # --- Feature Parameters ---
    vol_estimator_window: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Window for GK/Parkinson/YZ vol estimators.",
    )
    fractal_period: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Fractal dimension calculation period.",
    )
    hurst_window: int = Field(
        default=50,
        ge=20,
        le=200,
        description="Hurst exponent calculation window.",
    )
    er_period: int = Field(
        default=10,
        ge=5,
        le=50,
        description="Efficiency ratio calculation period.",
    )
    adx_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ADX calculation period.",
    )
    vov_window: int = Field(
        default=20,
        ge=5,
        le=60,
        description="Volatility-of-volatility window.",
    )

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=2190.0, gt=0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> Self:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            self.training_window
            + max(
                self.vol_estimator_window,
                self.fractal_period,
                self.hurst_window,
                self.vol_window,
            )
            + 10
        )
