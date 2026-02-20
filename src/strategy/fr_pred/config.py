"""FR-Pred (Funding Rate Prediction) 전략 설정.

FR z-score 평균회귀 + 모멘텀 이중 시그널.
극단 FR에서 contrarian 진입, FR 추세에서 momentum 추종.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class FRPredConfig(BaseModel):
    """FR-Pred (Funding Rate Prediction) 전략 설정.

    FR z-score 기반 이중 시그널:
    1. Mean-reversion: 극단 FR z-score → contrarian
    2. Momentum: FR MA crossover → trend-following

    Attributes:
        fr_ma_window: FR 이동평균 계산 window.
        fr_zscore_window: FR z-score rolling window.
        fr_mr_threshold: 평균회귀 진입 z-score 임계값.
        fr_mom_fast: FR momentum 빠른 MA window.
        fr_mom_slow: FR momentum 느린 MA window.
        mr_weight: 평균회귀 시그널 가중치 (0~1).
        mom_weight: 모멘텀 시그널 가중치 (0~1).
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 1D TF 연환산 계수.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준.
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- FR Parameters ---
    fr_ma_window: int = Field(default=7, ge=3, le=30)
    fr_zscore_window: int = Field(default=60, ge=20, le=200)
    fr_mr_threshold: float = Field(default=2.0, ge=0.5, le=4.0)

    # --- FR Momentum ---
    fr_mom_fast: int = Field(default=7, ge=3, le=30)
    fr_mom_slow: int = Field(default=21, ge=10, le=60)

    # --- Signal Weights ---
    mr_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    mom_weight: float = Field(default=0.5, ge=0.0, le=1.0)

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
    def _validate_cross_fields(self) -> FRPredConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.fr_mom_fast >= self.fr_mom_slow:
            msg = f"fr_mom_fast ({self.fr_mom_fast}) must be < fr_mom_slow ({self.fr_mom_slow})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.fr_zscore_window, self.fr_mom_slow, self.vol_window) + 10
