"""KAMA Efficiency Trend 전략 설정.

ER(Efficiency Ratio)이 높은 구간에서 KAMA 방향 추종.
정보 비대칭 기반, adaptive smoothing으로 whipsaw 방지.
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


class KamaEffTrendConfig(BaseModel):
    """KAMA Efficiency Trend 전략 설정.

    KAMA slope 방향 + ER threshold 필터로 추세 품질이 높은 구간만 진입.
    KAMA-price 거리를 conviction으로 활용하여 추세 강도에 비례한 포지션 사이징.

    Attributes:
        er_period: Efficiency Ratio 계산 기간.
        kama_period: KAMA ER lookback 기간.
        kama_fast: KAMA fast smoothing constant 기간.
        kama_slow: KAMA slow smoothing constant 기간.
        er_threshold: ER 진입 하한 (이상이면 시그널 활성).
        slope_window: KAMA slope 계산 rolling window.
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
    er_period: int = Field(default=10, ge=3, le=50, description="ER 계산 기간.")
    kama_period: int = Field(default=10, ge=3, le=50, description="KAMA ER lookback 기간.")
    kama_fast: int = Field(default=2, ge=2, le=10, description="KAMA fast SC 기간.")
    kama_slow: int = Field(default=30, ge=10, le=100, description="KAMA slow SC 기간.")
    er_threshold: float = Field(default=0.30, ge=0.05, le=0.80, description="ER 진입 하한 (0~1).")
    slope_window: int = Field(default=5, ge=2, le=20, description="KAMA slope 계산 rolling window.")

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
        if self.kama_slow <= self.kama_fast:
            msg = f"kama_slow ({self.kama_slow}) must be > kama_fast ({self.kama_fast})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            max(self.er_period, self.kama_period, self.kama_slow, self.vol_window, self.atr_period)
            + self.slope_window
            + 10
        )
