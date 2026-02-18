"""Vol Squeeze Breakout 전략 설정.

변동성 극저 스퀴즈 후 방향성 탈출 포착.
변동성 군집(GARCH) 기반 구조적 비효율, 이벤트 드리븐 극저빈도 거래.
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


class VolSqueezeBrkConfig(BaseModel):
    """Vol Squeeze Breakout 전략 설정.

    Attributes:
        bb_period: Bollinger Bands 기간.
        bb_std: Bollinger Bands 표준편차 배수.
        bb_pct_window: BB width 백분위 계산 윈도우 (~20일 at 4H).
        bb_pct_threshold: 스퀴즈 진입 백분위 기준 (이하면 squeeze).
        atr_period: ATR 계산 기간.
        atr_ratio_window: ATR ratio 장기 윈도우 (~15일 at 4H).
        atr_ratio_threshold: 단기/장기 ATR 비율 임계 (이하면 squeeze).
        vol_surge_window: 거래량 서지 기준 윈도우.
        vol_surge_multiplier: 거래량 서지 배수.
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 4H TF 연환산 계수.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY drawdown 기준.
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄.
    """

    model_config = ConfigDict(frozen=True)

    # --- Bollinger Bands ---
    bb_period: int = Field(default=20, ge=5, le=50)
    bb_std: float = Field(default=2.0, ge=1.0, le=4.0)
    bb_pct_window: int = Field(default=120, ge=30, le=500)
    bb_pct_threshold: float = Field(default=0.20, ge=0.05, le=0.50)

    # --- ATR Squeeze ---
    atr_period: int = Field(default=14, ge=5, le=50)
    atr_ratio_window: int = Field(default=90, ge=20, le=300)
    atr_ratio_threshold: float = Field(default=0.70, ge=0.3, le=1.0)

    # --- Volume Surge ---
    vol_surge_window: int = Field(default=42, ge=10, le=200)
    vol_surge_multiplier: float = Field(default=1.3, ge=1.0, le=3.0)

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
            max(self.bb_pct_window, self.atr_ratio_window, self.vol_surge_window, self.vol_window)
            + 10
        )
