"""Realized Semivariance Momentum 전략 설정.

일별 수익률의 양/음 MAD 기반 반분산 비율로 상방 vol 우위 모멘텀 감지.
Barndorff-Nielsen et al. (2010) + Skewness Risk & Crypto Returns (2024) 실증.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class UpVolMomConfig(BaseModel):
    """Realized Semivariance Momentum 전략 설정.

    Signal Formula:
        1. up_semivar = E[max(r, 0)^2] over rolling window
        2. down_semivar = E[min(r, 0)^2] over rolling window
        3. up_ratio = up_semivar / (up_semivar + down_semivar)
        4. ratio_ma = up_ratio.rolling(ratio_ma_window).mean()
        5. Long: ratio_ma > ratio_threshold + mom_return > 0
        6. Short: ratio_ma < (1 - ratio_threshold) + mom_return < 0

    Attributes:
        semivar_window: 반분산 계산 rolling window (일수).
        ratio_ma_window: 반분산 비율 이동평균 window.
        ratio_threshold: 상방 비율 진입 임계값 (>0.5 = upside 우위).
        mom_lookback: 모멘텀 방향 확인 lookback.
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
    semivar_window: int = Field(default=20, ge=5, le=120)
    ratio_ma_window: int = Field(default=10, ge=3, le=60)
    ratio_threshold: float = Field(default=0.55, gt=0.5, le=0.8)
    mom_lookback: int = Field(default=20, ge=5, le=120)

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
    def _validate_cross_fields(self) -> UpVolMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            max(self.semivar_window + self.ratio_ma_window, self.mom_lookback, self.vol_window) + 10
        )
