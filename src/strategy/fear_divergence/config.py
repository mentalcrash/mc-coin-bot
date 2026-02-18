"""Fear-Greed Divergence 전략 설정.

F&G 극단 + 가격 다이버전스 contrarian. 행동 편향 기반.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class FearDivergenceConfig(BaseModel):
    """Fear-Greed Divergence 전략 설정.

    Attributes:
        fg_fear_threshold: Fear 극단 임계값 (이하).
        fg_greed_threshold: Greed 극단 임계값 (이상).
        fg_ma_window: F&G smoothing window (4H bars, ~14일 = 84).
        fg_deviation: F&G vs MA 편차 임계값.
        price_roc_window: 가격 ROC window (4H bars).
        er_min: ER 최소 (추세 시작 확인).
        er_window: Efficiency Ratio window.
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 4H TF 연환산 계수.
        atr_period: ATR 계산 기간.
        short_mode: 숏 포지션 허용 모드.
    """

    model_config = ConfigDict(frozen=True)

    # --- F&G Parameters ---
    fg_fear_threshold: int = Field(default=20, ge=5, le=30)
    fg_greed_threshold: int = Field(default=80, ge=70, le=95)
    fg_ma_window: int = Field(default=84, ge=30, le=200)
    fg_deviation: float = Field(default=15.0, ge=5.0, le=30.0)

    # --- Price Divergence ---
    price_roc_window: int = Field(default=6, ge=3, le=18)

    # --- Trend Confirmation ---
    er_min: float = Field(default=0.2, ge=0.1, le=0.5)
    er_window: int = Field(default=15, ge=5, le=50)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=120)
    min_volatility: float = Field(default=0.05, gt=0.0, le=0.20)
    annualization_factor: float = Field(default=2190.0, gt=0.0)

    # --- ATR ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> FearDivergenceConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.fg_fear_threshold >= self.fg_greed_threshold:
            msg = (
                f"fg_fear_threshold ({self.fg_fear_threshold}) "
                f"must be < fg_greed_threshold ({self.fg_greed_threshold})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.fg_ma_window, self.er_window, self.vol_window) + 10
