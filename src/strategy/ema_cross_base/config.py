"""EMA Cross Base 설정 모델."""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class EmaCrossBaseConfig(BaseModel):
    """순수 20/100 EMA 크로스오버 설정.

    EMA(fast) > EMA(slow) → Long, EMA(fast) < EMA(slow) → Short.
    """

    model_config = ConfigDict(frozen=True)

    fast_period: int = Field(default=20, ge=5, le=50)
    slow_period: int = Field(default=100, ge=50, le=300)

    vol_window: int = Field(default=30, ge=5, le=252)
    vol_target: float = Field(default=0.35, ge=0.05, le=1.0)
    min_volatility: float = Field(default=0.01, ge=0.001, le=0.1)
    annualization_factor: float = Field(default=365.0, ge=1.0)

    short_mode: ShortMode = Field(default=ShortMode.FULL)

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return self.slow_period + self.vol_window + 10
