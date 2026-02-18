"""Liquidation Cascade Reversal 전략 설정.

레버리지 캐스케이드 후 가격 오버슈트 평균회귀.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class LiqCascadeRevConfig(BaseModel):
    """Liquidation Cascade Reversal 전략 설정.

    Attributes:
        fr_zscore_window: FR z-score rolling window.
        fr_ma_window: FR MA window.
        fr_buildup_threshold: 스트레스 축적 기준 (|fr_z|).
        cascade_return_multiplier: |return| > N x ATR 캐스케이드 판정.
        vol_expansion_ratio: 단기vol/장기vol 캐스케이드 확인.
        vol_short_window: 단기 변동성 window.
        vol_long_window: 장기 변동성 window.
        reversal_confirmation_bars: 리버설 확인 대기 bar 수.
        recovery_body_pct: 캔들 body 회복 비율 임계값.
        max_hold_bars: 최대 보유 기간 (4H bars).
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 4H TF 연환산 계수.
        atr_period: ATR 계산 기간.
        short_mode: 숏 포지션 허용 모드.
    """

    model_config = ConfigDict(frozen=True)

    # --- FR Parameters ---
    fr_zscore_window: int = Field(default=42, ge=20, le=200)
    fr_ma_window: int = Field(default=9, ge=3, le=30)
    fr_buildup_threshold: float = Field(default=2.0, ge=1.0, le=4.0)

    # --- Cascade Detection ---
    cascade_return_multiplier: float = Field(default=2.5, ge=1.5, le=5.0)
    vol_expansion_ratio: float = Field(default=2.0, ge=1.3, le=4.0)
    vol_short_window: int = Field(default=6, ge=3, le=20)
    vol_long_window: int = Field(default=42, ge=20, le=120)

    # --- Reversal Confirmation ---
    reversal_confirmation_bars: int = Field(default=1, ge=0, le=3)
    recovery_body_pct: float = Field(default=0.5, ge=0.2, le=0.9)

    # --- Hold Limit ---
    max_hold_bars: int = Field(default=18, ge=6, le=42)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=120)
    min_volatility: float = Field(default=0.05, gt=0.0, le=0.20)
    annualization_factor: float = Field(default=2190.0, gt=0.0)

    # --- ATR ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> LiqCascadeRevConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.vol_short_window >= self.vol_long_window:
            msg = (
                f"vol_short_window ({self.vol_short_window}) "
                f"must be < vol_long_window ({self.vol_long_window})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.fr_zscore_window, self.vol_long_window, self.vol_window) + 10
