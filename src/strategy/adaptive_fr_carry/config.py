"""Adaptive FR Carry 전략 설정.

FR 극단 구간에서만 캐리 수취 + vol 필터로 캐스케이드 회피.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class AdaptiveFrCarryConfig(BaseModel):
    """Adaptive FR Carry 전략 설정.

    Attributes:
        fr_ma_window: Funding rate MA window.
        fr_zscore_window: FR z-score rolling window (4H bars).
        fr_entry_threshold: FR z-score 진입 임계값 (절대값).
        fr_exit_threshold: FR z-score 청산 임계값 (절대값).
        vol_ratio_exit: ATR ratio 상한 (캐스케이드 회피).
        er_max: Efficiency Ratio 상한 (추세 배제).
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 4H TF 연환산 계수.
        atr_period: ATR 계산 기간.
        atr_short_window: 단기 ATR rolling window.
        atr_long_window: 장기 ATR rolling window.
        max_hold_bars: 최대 보유 기간 (4H bars).
        short_mode: 숏 포지션 허용 모드.
    """

    model_config = ConfigDict(frozen=True)

    # --- FR Parameters ---
    fr_ma_window: int = Field(default=9, ge=3, le=30)
    fr_zscore_window: int = Field(default=42, ge=20, le=200)
    fr_entry_threshold: float = Field(default=2.0, ge=1.0, le=4.0)
    fr_exit_threshold: float = Field(default=1.0, ge=0.0, le=2.0)

    # --- Vol Filter ---
    vol_ratio_exit: float = Field(default=1.5, ge=1.0, le=3.0)
    er_max: float = Field(default=0.4, ge=0.1, le=0.8)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=120)
    min_volatility: float = Field(default=0.05, gt=0.0, le=0.20)
    annualization_factor: float = Field(default=2190.0, gt=0.0)

    # --- ATR Parameters ---
    atr_period: int = Field(default=14, ge=5, le=50)
    atr_short_window: int = Field(default=6, ge=3, le=20)
    atr_long_window: int = Field(default=42, ge=20, le=120)

    # --- Hold Limit ---
    max_hold_bars: int = Field(default=42, ge=10, le=100)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> AdaptiveFrCarryConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.fr_exit_threshold >= self.fr_entry_threshold:
            msg = (
                f"fr_exit_threshold ({self.fr_exit_threshold}) "
                f"must be < fr_entry_threshold ({self.fr_entry_threshold})"
            )
            raise ValueError(msg)
        if self.atr_short_window >= self.atr_long_window:
            msg = (
                f"atr_short_window ({self.atr_short_window}) "
                f"must be < atr_long_window ({self.atr_long_window})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.fr_zscore_window, self.vol_window, self.atr_long_window) + 10
