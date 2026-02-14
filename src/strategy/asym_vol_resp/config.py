"""Asymmetric Volume Response 전략 설정.

Kyle(1985) informed trading model.
Volume-price impact ratio로 informed flow 감지.
Volume이 price에 미치는 impact 비대칭 분석.
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


class AsymVolRespConfig(BaseModel):
    """Asymmetric Volume Response 전략 설정.

    Attributes:
        impact_window: Volume-price impact 계산 윈도우 (bars).
        asym_window: Asymmetry rolling 계산 윈도우 (bars).
        asym_threshold: |asymmetry_zscore| > threshold → informed flow.
        mom_lookback: Momentum direction lookback (bars).
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

    # --- Impact Parameters ---
    impact_window: int = Field(
        default=12,
        ge=3,
        le=60,
        description="Volume-price impact calculation window.",
    )
    asym_window: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Asymmetry rolling calculation window.",
    )
    asym_threshold: float = Field(
        default=1.0,
        ge=0.3,
        le=3.0,
        description="|asymmetry_zscore| > threshold = informed flow.",
    )

    # --- Momentum Direction ---
    mom_lookback: int = Field(
        default=12,
        ge=3,
        le=60,
        description="Momentum direction lookback.",
    )

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=2190.0, gt=0)

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
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            max(
                self.impact_window + self.asym_window,
                self.mom_lookback,
                self.vol_window,
                self.atr_period,
            )
            + 10
        )
