"""Funding Rate + Stablecoin Confluence 전략 설정.

FR 극단값과 Stablecoin flow 확인을 결합하여
과열/과냉 반전 포인트를 포착한다.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class FrStabConfConfig(BaseModel):
    """Funding Rate + Stablecoin Confluence 전략 설정.

    Attributes:
        fr_ma_window: Funding rate MA window.
        fr_zscore_window: FR z-score rolling window (4H bars).
        fr_short_threshold: Short 진입 FR z-score (양수 극단).
        fr_long_threshold: Long 진입 FR z-score (음수 극단).
        fr_exit_threshold: 포지션 청산 FR z-score 임계값.
        stab_change_period: Stablecoin pct_change lookback (4H bars).
        stab_zscore_window: Stablecoin z-score rolling window.
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 4H TF 연환산 계수 (2190).
        atr_period: ATR 계산 기간.
        short_mode: 숏 포지션 허용 모드.
    """

    model_config = ConfigDict(frozen=True)

    # --- FR Parameters ---
    fr_ma_window: int = Field(default=9, ge=3, le=30)
    fr_zscore_window: int = Field(default=540, ge=60, le=1500)
    fr_short_threshold: float = Field(default=2.0, ge=1.0, le=5.0)
    fr_long_threshold: float = Field(default=-1.5, ge=-5.0, le=-0.5)
    fr_exit_threshold: float = Field(default=0.5, ge=0.0, le=2.0)

    # --- Stablecoin Parameters ---
    stab_change_period: int = Field(default=42, ge=6, le=180)
    stab_zscore_window: int = Field(default=540, ge=60, le=1500)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=120)
    min_volatility: float = Field(default=0.05, gt=0.0, le=0.20)
    annualization_factor: float = Field(default=2190.0, gt=0.0)

    # --- Options ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> FrStabConfConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.fr_long_threshold >= -self.fr_exit_threshold:
            msg = (
                f"fr_long_threshold ({self.fr_long_threshold}) "
                f"must be < -fr_exit_threshold ({-self.fr_exit_threshold})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.fr_zscore_window, self.stab_zscore_window + self.stab_change_period) + 10
