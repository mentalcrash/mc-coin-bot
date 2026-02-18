"""Vol Squeeze + Derivatives 전략 설정.

Vol percentile rank 압축 → breakout + FR 방향 확인.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class VolSqueezeDerivConfig(BaseModel):
    """Vol Squeeze + Derivatives 전략 설정.

    Attributes:
        vol_rank_window: Vol percentile rank 기준 기간 (4H bars).
        squeeze_threshold: 하위 N percentile = 압축 판정.
        min_squeeze_bars: 최소 압축 지속 bar 수.
        expansion_ratio: ATR ratio for expansion 판정.
        fr_zscore_window: FR z-score rolling window.
        fr_ma_window: FR MA window.
        sma_direction_window: 가격 방향 판정 SMA window.
        contrarian_weight: FR 반대 방향 weight (carry 수취).
        aligned_weight: FR 같은 방향 weight.
        vol_exit_rank: Vol 과확장 → 청산 percentile.
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 4H TF 연환산 계수.
        atr_period: ATR 계산 기간.
        atr_short_window: 단기 ATR rolling window.
        atr_long_window: 장기 ATR rolling window.
        short_mode: 숏 포지션 허용 모드.
    """

    model_config = ConfigDict(frozen=True)

    # --- Squeeze Parameters ---
    vol_rank_window: int = Field(default=126, ge=30, le=504)
    squeeze_threshold: int = Field(default=15, ge=5, le=30)
    min_squeeze_bars: int = Field(default=3, ge=1, le=10)
    expansion_ratio: float = Field(default=1.3, ge=1.1, le=2.0)

    # --- FR Parameters ---
    fr_zscore_window: int = Field(default=42, ge=20, le=200)
    fr_ma_window: int = Field(default=9, ge=3, le=30)

    # --- Direction ---
    sma_direction_window: int = Field(default=15, ge=5, le=50)
    contrarian_weight: float = Field(default=1.0, ge=0.1, le=2.0)
    aligned_weight: float = Field(default=0.6, ge=0.1, le=1.5)

    # --- Exit ---
    vol_exit_rank: int = Field(default=70, ge=50, le=90)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=120)
    min_volatility: float = Field(default=0.05, gt=0.0, le=0.20)
    annualization_factor: float = Field(default=2190.0, gt=0.0)

    # --- ATR ---
    atr_period: int = Field(default=14, ge=5, le=50)
    atr_short_window: int = Field(default=6, ge=3, le=20)
    atr_long_window: int = Field(default=42, ge=20, le=120)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> VolSqueezeDerivConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
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
        return max(self.vol_rank_window, self.fr_zscore_window, self.atr_long_window) + 10
