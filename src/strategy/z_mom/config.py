"""Z-Momentum (MACD-V) 전략 설정.

ATR-정규화 MACD (MACD-V) + 명시적 flat zone으로 Vol-anchoring bias 교정.
Spiroglou(2022) NAAIM/CMT 이중 수상 방법론 기반.
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


class ZMomConfig(BaseModel):
    """Z-Momentum (MACD-V) 전략 설정.

    MACD-V = MACD_line / ATR 로 변동성 정규화하여
    고변동 구간에서의 과대 시그널을 억제한다.
    flat_zone 내 MACD-V 값은 중립(0) 처리하여 노이즈 필터링.

    Attributes:
        macd_fast: MACD fast EMA 기간.
        macd_slow: MACD slow EMA 기간.
        macd_signal: MACD signal line EMA 기간.
        atr_period: ATR 계산 기간 (MACD-V 정규화용).
        flat_zone: MACD-V flat zone 임계값 (절대값 이내 = 중립).
        mom_lookback: 모멘텀 확인 lookback (bars).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 12H = 730.0.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    macd_fast: int = Field(default=12, ge=3, le=30, description="MACD fast EMA period.")
    macd_slow: int = Field(default=26, ge=10, le=60, description="MACD slow EMA period.")
    macd_signal: int = Field(default=9, ge=3, le=30, description="MACD signal line EMA period.")
    atr_period: int = Field(
        default=14, ge=5, le=50, description="ATR period for MACD-V normalization."
    )
    flat_zone: float = Field(
        default=0.3,
        ge=0.0,
        le=5.0,
        description="MACD-V flat zone threshold (absolute). Values within are neutral.",
    )
    mom_lookback: int = Field(
        default=5,
        ge=3,
        le=60,
        description="Momentum confirmation lookback (bars).",
    )

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.50, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=730.0, gt=0)

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
        if self.macd_slow <= self.macd_fast:
            msg = f"macd_slow ({self.macd_slow}) must be > macd_fast ({self.macd_fast})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            max(
                self.macd_slow + self.macd_signal,
                self.atr_period,
                self.mom_lookback,
                self.vol_window,
            )
            + 10
        )
