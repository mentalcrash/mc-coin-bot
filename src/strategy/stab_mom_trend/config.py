"""Stablecoin Momentum Trend 전략 설정.

Stablecoin 공급 변화율 z-score와 4H EMA cross를 결합하여
자금 유입/유출 모멘텀을 포착한다.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class StabMomTrendConfig(BaseModel):
    """Stablecoin Momentum Trend 전략 설정.

    Attributes:
        stab_change_period: Stablecoin pct_change lookback (4H bars, 42=7일).
        zscore_window: Rolling z-score window (4H bars, 540=90일).
        stab_long_threshold: Long 진입 z-score 임계값.
        stab_short_threshold: Short 진입 z-score 임계값.
        ema_fast_period: EMA fast span (4H bars).
        ema_slow_period: EMA slow span (4H bars).
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 4H TF 연환산 계수 (2190).
        atr_period: ATR 계산 기간.
        short_mode: 숏 포지션 허용 모드.
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Stablecoin Parameters ---
    stab_change_period: int = Field(default=42, ge=6, le=180)
    zscore_window: int = Field(default=540, ge=60, le=1500)
    stab_long_threshold: float = Field(default=0.5, ge=0.0, le=3.0)
    stab_short_threshold: float = Field(default=-0.5, ge=-3.0, le=0.0)

    # --- EMA Parameters ---
    ema_fast_period: int = Field(default=24, ge=6, le=120)
    ema_slow_period: int = Field(default=72, ge=24, le=360)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=120)
    min_volatility: float = Field(default=0.05, gt=0.0, le=0.20)
    annualization_factor: float = Field(default=2190.0, gt=0.0)

    # --- Options ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_strength_ratio: float = Field(default=0.5, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> StabMomTrendConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.ema_fast_period >= self.ema_slow_period:
            msg = f"ema_fast_period ({self.ema_fast_period}) must be < ema_slow_period ({self.ema_slow_period})"
            raise ValueError(msg)
        if self.stab_short_threshold >= self.stab_long_threshold:
            msg = (
                f"stab_short_threshold ({self.stab_short_threshold}) "
                f"must be < stab_long_threshold ({self.stab_long_threshold})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return self.zscore_window + self.stab_change_period + 10
