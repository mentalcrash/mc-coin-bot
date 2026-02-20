"""F&G EMA Long-Cycle 전략 설정.

F&G EMA-24w가 장기 사이클 최강 예측변수 (arXiv:2512.02029).
장기 센티먼트 추세의 크로스오버로 매크로 사이클 타이밍.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class FgEmaCycleConfig(BaseModel):
    """F&G EMA Long-Cycle 전략 설정.

    Attributes:
        ema_slow_span: 장기 EMA 기간 (24주 ≈ 168일).
        ema_fast_span: 단기 EMA 기간 (6주 ≈ 42일).
        fear_cycle: 장기 EMA가 이 값 이하일 때 long 시그널 유효.
        greed_cycle: 장기 EMA가 이 값 이상일 때 short 시그널 유효.
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 1D TF 연환산 계수.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준.
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    ema_slow_span: int = Field(default=168, ge=60, le=365)
    ema_fast_span: int = Field(default=42, ge=10, le=120)
    fear_cycle: float = Field(default=35.0, ge=10.0, le=50.0)
    greed_cycle: float = Field(default=65.0, ge=50.0, le=90.0)

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
    def _validate_cross_fields(self) -> FgEmaCycleConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.ema_fast_span >= self.ema_slow_span:
            msg = f"ema_fast_span ({self.ema_fast_span}) must be < ema_slow_span ({self.ema_slow_span})"
            raise ValueError(msg)
        if self.fear_cycle >= self.greed_cycle:
            msg = f"fear_cycle ({self.fear_cycle}) must be < greed_cycle ({self.greed_cycle})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return self.ema_slow_span + 20
