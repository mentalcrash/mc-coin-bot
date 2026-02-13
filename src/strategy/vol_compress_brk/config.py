"""Vol-Compression Breakout 전략 설정.

ATR compression(ratio<0.6) 후 expansion(ratio>1.2) 전이 감지 시
첫 방향성 이동이 breakout으로 지속. GARCH volatility clustering 기반.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class VolCompressBrkConfig(BaseModel):
    """Vol-Compression Breakout 전략 설정.

    Attributes:
        atr_fast: Fast ATR period.
        atr_slow: Slow ATR period.
        compress_threshold: Compression 임계값 (ATR fast/slow ratio).
        expand_threshold: Expansion 임계값 (ATR fast/slow ratio).
        mom_lookback: Breakout 방향 결정용 momentum lookback.
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 15m TF 연환산 계수 (35040).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    atr_fast: int = Field(default=7, ge=3, le=20)
    atr_slow: int = Field(default=28, ge=10, le=100)
    compress_threshold: float = Field(default=0.6, gt=0.0, le=1.0)
    expand_threshold: float = Field(default=1.2, ge=1.0, le=3.0)
    mom_lookback: int = Field(default=10, ge=3, le=50)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=35040.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> VolCompressBrkConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.atr_fast >= self.atr_slow:
            msg = f"atr_fast ({self.atr_fast}) must be < atr_slow ({self.atr_slow})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.atr_slow, self.mom_lookback, self.vol_window) + 10
