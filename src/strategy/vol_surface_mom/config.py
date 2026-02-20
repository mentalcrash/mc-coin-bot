"""Volatility Surface Momentum 전략 설정.

GK/YZ/Parkinson vol 비율이 시장 미시구조 정보를 인코딩.
CTREND 미사용 지표 전량 활용. 학술근거: SSRN 5048674, FRL 2025.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class VolSurfaceMomConfig(BaseModel):
    """Volatility Surface Momentum 전략 설정.

    GK, Parkinson, Yang-Zhang 세 가지 range-based vol 추정치의 비율로
    시장의 미시구조 변화를 감지하여 모멘텀 시그널을 생성합니다.

    Attributes:
        gk_window: GK vol rolling window.
        pk_window: Parkinson vol rolling window.
        yz_window: Yang-Zhang vol rolling window.
        ratio_window: Vol ratio smoothing window.
        momentum_window: 가격 모멘텀 확인 기간 (ROC).
        gk_pk_long_threshold: GK/PK ratio long 임계값 (>1 = close vol 우세).
        gk_pk_short_threshold: GK/PK ratio short 임계값 (<1 = range vol 우세).
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
    gk_window: int = Field(default=21, ge=5, le=100)
    pk_window: int = Field(default=21, ge=5, le=100)
    yz_window: int = Field(default=21, ge=5, le=100)
    ratio_window: int = Field(default=14, ge=3, le=60)
    momentum_window: int = Field(default=21, ge=5, le=60)
    gk_pk_long_threshold: float = Field(default=1.05, ge=0.80, le=1.50)
    gk_pk_short_threshold: float = Field(default=0.95, ge=0.50, le=1.20)

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
    def _validate_cross_fields(self) -> VolSurfaceMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.gk_pk_long_threshold <= self.gk_pk_short_threshold:
            msg = (
                f"gk_pk_long_threshold ({self.gk_pk_long_threshold}) must be > "
                f"gk_pk_short_threshold ({self.gk_pk_short_threshold})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            max(
                self.gk_window,
                self.pk_window,
                self.yz_window,
                self.momentum_window,
                self.vol_window,
            )
            + self.ratio_window
            + 10
        )
