"""Multi-Scale Volatility Ratio 전략 설정.

Vol term structure inversion = 정보 도착 프록시.
단기vol/장기vol ratio로 breakout timing 포착.
Corsi(2009) HAR-RV 이론 기반.
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


class MSVolRatioConfig(BaseModel):
    """Multi-Scale Volatility Ratio 전략 설정.

    Attributes:
        short_vol_window: 단기 변동성 윈도우 (bars).
        long_vol_window: 장기 변동성 윈도우 (bars).
        ratio_smooth: Vol ratio EMA smoothing 기간.
        ratio_upper: Ratio > upper → breakout 감지.
        ratio_lower: Ratio < lower → calm regime (no trade).
        mom_lookback: Momentum direction lookback.
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 4H = 2190.0.
        atr_period: ATR 계산 기간.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY drawdown 기준.
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄.
    """

    model_config = ConfigDict(frozen=True)

    # --- Vol Ratio Parameters ---
    short_vol_window: int = Field(
        default=6,
        ge=3,
        le=30,
        description="Short-term volatility window (bars).",
    )
    long_vol_window: int = Field(
        default=48,
        ge=20,
        le=200,
        description="Long-term volatility window (bars).",
    )
    ratio_smooth: int = Field(
        default=6,
        ge=2,
        le=30,
        description="EMA smoothing for vol ratio.",
    )
    ratio_upper: float = Field(
        default=1.3,
        ge=0.8,
        le=3.0,
        description="Upper ratio threshold (information arrival).",
    )
    ratio_lower: float = Field(
        default=0.7,
        ge=0.2,
        le=1.5,
        description="Lower ratio threshold (calm, no trade).",
    )

    # --- Momentum Direction ---
    mom_lookback: int = Field(
        default=12,
        ge=3,
        le=60,
        description="Momentum direction lookback (bars).",
    )

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=2190.0, gt=0)

    # --- Options ---
    atr_period: int = Field(default=14, ge=5, le=50)

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
        if self.short_vol_window >= self.long_vol_window:
            msg = (
                f"short_vol_window ({self.short_vol_window}) must be < "
                f"long_vol_window ({self.long_vol_window})"
            )
            raise ValueError(msg)
        if self.ratio_upper <= self.ratio_lower:
            msg = f"ratio_upper ({self.ratio_upper}) must be > ratio_lower ({self.ratio_lower})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.long_vol_window, self.vol_window, self.atr_period) + self.ratio_smooth + 10
