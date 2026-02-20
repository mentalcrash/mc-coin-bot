"""Disposition Breakout 전략 설정.

처분 효과(disposition effect) - 투자자가 rolling high 근처에서 매도 압력 생성.
돌파 시 후회 회피(regret aversion)로 추격 매수 발생. George-Hwang(2004).
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class DispBreakoutConfig(BaseModel):
    """Disposition Breakout 전략 설정.

    Attributes:
        high_window: Rolling high 계산 기간.
        proximity_threshold: 고점 근접 비율 (close / rolling_high >= threshold).
        breakout_threshold: 돌파 확인 비율 (close / rolling_high >= threshold).
        momentum_window: 모멘텀 확인 기간.
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 1D TF 연환산 계수.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    high_window: int = Field(default=52, ge=10, le=252)
    proximity_threshold: float = Field(default=0.95, ge=0.80, le=1.0)
    breakout_threshold: float = Field(default=1.0, ge=0.95, le=1.10)
    momentum_window: int = Field(default=21, ge=5, le=60)

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
    def _validate_cross_fields(self) -> DispBreakoutConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.high_window, self.momentum_window, self.vol_window) + 10
