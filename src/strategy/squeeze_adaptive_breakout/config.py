"""Squeeze-Adaptive Breakout 전략 설정.

BB inside KC (squeeze) 해제 시점에 KAMA 적응적 방향 + BB position 확인으로
레버리지 캐스케이드 과대 이동을 포착하는 전략.
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


class SqueezeAdaptiveBreakoutConfig(BaseModel):
    """Squeeze-Adaptive Breakout 전략 설정.

    Attributes:
        bb_period: Bollinger Band 계산 기간.
        bb_std: BB 표준편차 배수.
        kc_period: Keltner Channel EMA 기간.
        kc_atr_period: Keltner Channel ATR 기간.
        kc_mult: KC ATR 배수.
        kama_er_lookback: KAMA Efficiency Ratio 룩백.
        kama_fast: KAMA 빠른 smoothing 기간.
        kama_slow: KAMA 느린 smoothing 기간.
        bb_pos_period: BB position 계산 기간.
        bb_pos_std: BB position 표준편차 배수.
        bb_pos_long_threshold: BB position long 진입 기준 (상위).
        bb_pos_short_threshold: BB position short 진입 기준 (하위).
        squeeze_lookback: 연속 squeeze bar 최소 수.
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: TF별 연환산 계수 (1D=365).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Bollinger Bands ---
    bb_period: int = Field(default=20, ge=5, le=50)
    bb_std: float = Field(default=2.0, ge=1.0, le=3.0)

    # --- Keltner Channels ---
    kc_period: int = Field(default=20, ge=5, le=50)
    kc_atr_period: int = Field(default=10, ge=5, le=50)
    kc_mult: float = Field(default=1.5, ge=0.5, le=3.0)

    # --- KAMA (Adaptive Direction) ---
    kama_er_lookback: int = Field(default=10, ge=3, le=30)
    kama_fast: int = Field(default=2, ge=2, le=10)
    kama_slow: int = Field(default=30, ge=10, le=60)

    # --- BB Position (Conviction) ---
    bb_pos_period: int = Field(default=20, ge=5, le=50)
    bb_pos_std: float = Field(default=2.0, ge=1.0, le=3.0)
    bb_pos_long_threshold: float = Field(default=0.7, ge=0.5, le=1.0)
    bb_pos_short_threshold: float = Field(default=0.3, ge=0.0, le=0.5)

    # --- Squeeze Duration ---
    squeeze_lookback: int = Field(default=3, ge=1, le=20)

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
            max(
                self.bb_period,
                self.kc_period,
                self.kc_atr_period,
                self.kama_er_lookback + self.kama_slow,
                self.bb_pos_period,
                self.vol_window,
            )
            + self.squeeze_lookback
            + 10
        )
