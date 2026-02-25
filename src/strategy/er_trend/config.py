"""ER Trend 전략 설정.

Multi-lookback Efficiency Ratio x direction으로 추세 품질과 방향을 동시 포착.
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


class ErTrendConfig(BaseModel):
    """ER Trend 전략 설정.

    Multi-lookback Signed ER (ER * sign(direction))을 가중 합성하여
    추세 품질과 방향을 동시에 포착한다.

    Attributes:
        er_fast: 단기 ER 계산 기간.
        er_mid: 중기 ER 계산 기간.
        er_slow: 장기 ER 계산 기간.
        w_fast: 단기 ER 가중치.
        w_mid: 중기 ER 가중치.
        w_slow: 장기 ER 가중치.
        entry_threshold: 복합 signed ER 진입/청산 임계값.
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 12H TF 연환산 계수 (730.0).
        atr_period: ATR 계산 기간.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    er_fast: int = Field(default=10, ge=3, le=30, description="단기 ER 기간.")
    er_mid: int = Field(default=21, ge=10, le=60, description="중기 ER 기간.")
    er_slow: int = Field(default=42, ge=20, le=120, description="장기 ER 기간.")
    w_fast: float = Field(default=0.25, ge=0.0, le=1.0, description="단기 ER 가중치.")
    w_mid: float = Field(default=0.50, ge=0.0, le=1.0, description="중기 ER 가중치.")
    w_slow: float = Field(default=0.25, ge=0.0, le=1.0, description="장기 ER 가중치.")
    entry_threshold: float = Field(default=0.15, ge=0.01, le=0.8, description="진입 임계값.")

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
    def _validate_cross_fields(self) -> Self:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if not (self.er_fast < self.er_mid < self.er_slow):
            msg = (
                f"ER periods must be strictly increasing: "
                f"er_fast ({self.er_fast}) < er_mid ({self.er_mid}) < er_slow ({self.er_slow})"
            )
            raise ValueError(msg)
        total_w = self.w_fast + self.w_mid + self.w_slow
        weight_tolerance = 1e-6
        if abs(total_w - 1.0) > weight_tolerance:
            msg = f"Weights must sum to 1.0, got {total_w}"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.er_slow, self.vol_window, self.atr_period) + 10
