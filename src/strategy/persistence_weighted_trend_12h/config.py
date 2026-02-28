"""Persistence-Weighted-Trend 12H 전략 설정.

ER x FD x TrendStrength 복합 추세 품질 점수로 모멘텀 가중.
지속성 높은 추세만 진입하여 whipsaw 필터링.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class PersistenceWeightedTrendConfig(BaseModel):
    """Persistence-Weighted-Trend 12H 전략 설정.

    Attributes:
        scale_short: 단기 lookback (bars).
        scale_mid: 중기 lookback (bars).
        scale_long: 장기 lookback (bars).
        persistence_threshold: 진입 최소 persistence score (0~1).
        mom_lookback: 모멘텀 방향 ROC lookback (bars).
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

    # --- Multi-Scale Parameters ---
    scale_short: int = Field(default=14, ge=5, le=50)
    scale_mid: int = Field(default=30, ge=10, le=100)
    scale_long: int = Field(default=60, ge=20, le=200)

    # --- Persistence ---
    persistence_threshold: float = Field(default=0.3, ge=0.1, le=0.8)
    mom_lookback: int = Field(default=20, ge=5, le=100)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=730.0, gt=0.0)

    # --- Options ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> PersistenceWeightedTrendConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if not (self.scale_short < self.scale_mid < self.scale_long):
            msg = f"scale_short < scale_mid < scale_long required: {self.scale_short} < {self.scale_mid} < {self.scale_long}"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.scale_long, self.vol_window, self.mom_lookback) + 10
