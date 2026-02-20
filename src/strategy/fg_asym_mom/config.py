"""F&G Asymmetric Momentum 전략 설정.

Fear=역발상(contrarian), Greed=순응(momentum) 비대칭 접근.
FRL 2024 학술 실증: Fear herding > Greed herding.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class FgAsymMomConfig(BaseModel):
    """F&G Asymmetric Momentum 전략 설정.

    Attributes:
        fear_threshold: Fear zone 임계값 (F&G < threshold).
        greed_threshold: Greed extreme 임계값 (숏 시그널용).
        greed_hold_threshold: Greed hold 임계값 (모멘텀 유지).
        neutral_low: 중립 구간 하한 (exit zone).
        neutral_high: 중립 구간 상한 (exit zone).
        sma_short: 단기 SMA 기간 (가격 확인용).
        sma_long: 장기 SMA 기간 (모멘텀 확인용).
        fg_delta_window: F&G 변화율 윈도우.
        greed_persist_min: Greed extreme 최소 체류 기간 (숏 시그널용).
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 1D TF 연환산 계수.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준.
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Fear-side (Contrarian) ---
    fear_threshold: float = Field(default=25.0, ge=5.0, le=50.0)
    sma_short: int = Field(default=10, ge=3, le=30)
    fg_delta_window: int = Field(default=5, ge=2, le=15)

    # --- Greed-side (Momentum + Short) ---
    greed_threshold: float = Field(default=75.0, ge=60.0, le=95.0)
    greed_hold_threshold: float = Field(default=55.0, ge=40.0, le=70.0)
    sma_long: int = Field(default=20, ge=10, le=60)
    greed_persist_min: int = Field(default=5, ge=2, le=20)

    # --- Neutral zone (exit) ---
    neutral_low: float = Field(default=40.0, ge=20.0, le=50.0)
    neutral_high: float = Field(default=60.0, ge=50.0, le=80.0)

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
    def _validate_cross_fields(self) -> FgAsymMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.fear_threshold >= self.greed_hold_threshold:
            msg = (
                f"fear_threshold ({self.fear_threshold}) must be "
                f"< greed_hold_threshold ({self.greed_hold_threshold})"
            )
            raise ValueError(msg)
        if self.neutral_low >= self.neutral_high:
            msg = f"neutral_low ({self.neutral_low}) must be < neutral_high ({self.neutral_high})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.sma_long, self.vol_window, self.fg_delta_window) + 10
