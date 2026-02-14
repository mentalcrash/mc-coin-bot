"""Keltner Efficiency Trend 전략 설정.

ATR 기반 Keltner Channel 돌파 + Efficiency Ratio 확인으로 고품질 추세 진입.
KC는 vol 적응형 밴드, ER은 추세 품질 게이트.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class KeltEffTrendConfig(BaseModel):
    """Keltner Efficiency Trend 전략 설정.

    Attributes:
        kc_ema_period: Keltner Channel EMA 기간.
        kc_atr_period: Keltner Channel ATR 기간.
        kc_multiplier: Keltner Channel ATR 배수.
        er_period: Efficiency Ratio 계산 기간.
        er_threshold: ER 최소 임계값 (이상일 때만 진입).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 4H TF 연환산 계수 (2190).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    kc_ema_period: int = Field(default=20, ge=5, le=100)
    kc_atr_period: int = Field(default=10, ge=3, le=50)
    kc_multiplier: float = Field(default=1.5, ge=0.5, le=5.0)
    er_period: int = Field(default=10, ge=3, le=50)
    er_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=2190.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> KeltEffTrendConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.kc_ema_period, self.kc_atr_period, self.er_period, self.vol_window) + 10
