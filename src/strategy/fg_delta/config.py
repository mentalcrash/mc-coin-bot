"""Fear & Greed Delta 전략 설정.

F&G Index의 수준이 아닌 변화율(delta)을 시그널로 활용.
센티먼트 모멘텀이 가격 방향의 선행 지표.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class FgDeltaConfig(BaseModel):
    """Fear & Greed Delta 전략 설정.

    Attributes:
        fg_delta_window: F&G 변화율 계산 기간.
        fg_smooth_window: F&G delta smoothing 기간.
        delta_threshold: 시그널 발생 delta 임계값.
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
    fg_delta_window: int = Field(default=5, ge=1, le=30)
    fg_smooth_window: int = Field(default=7, ge=3, le=30)
    delta_threshold: float = Field(default=5.0, ge=0.0, le=30.0)

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
    def _validate_cross_fields(self) -> FgDeltaConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.fg_delta_window + self.fg_smooth_window, self.vol_window) + 10
