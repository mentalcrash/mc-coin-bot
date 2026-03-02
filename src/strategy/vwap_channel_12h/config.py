"""VWAP-Channel Multi-Scale 전략 설정.

VWAP 기반 3-scale channel breakout consensus.
Volume-weighted 중심선+밴드로 거래량이 동의한 breakout만 포착.
Multi-scale(20/60/150) consensus voting으로 noise rejection.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class VwapChannelConfig(BaseModel):
    """VWAP-Channel Multi-Scale 전략 설정.

    VWAP(Volume-Weighted Average Price) 기반 3-scale channel breakout 앙상블.
    각 스케일에서 rolling VWAP + ATR-based band를 구성하고,
    3개 스케일의 breakout consensus로 direction을 결정한다.

    Signal Logic:
        1. 각 스케일(short/mid/long)에서 VWAP 중심선 + band 계산
        2. close vs VWAP band breakout sub-signal 계산 (총 3개)
        3. consensus = mean(signal_short, signal_mid, signal_long)
        4. |consensus| >= entry_threshold -> direction = sign(consensus)
        5. strength = |consensus| * vol_scalar

    Attributes:
        scale_short: 단기 VWAP lookback (bars).
        scale_mid: 중기 VWAP lookback (bars).
        scale_long: 장기 VWAP lookback (bars).
        band_multiplier: ATR 기반 band 배수.
        atr_period: ATR 계산 기간.
        entry_threshold: consensus 진입 임계값 (0~1).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 12H TF 연환산 계수 (730).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    scale_short: int = Field(default=20, ge=5, le=100)
    scale_mid: int = Field(default=60, ge=10, le=300)
    scale_long: int = Field(default=150, ge=20, le=500)
    band_multiplier: float = Field(
        default=2.0,
        ge=0.5,
        le=5.0,
        description="ATR 기반 VWAP band 배수",
    )
    atr_period: int = Field(default=14, ge=5, le=100)
    entry_threshold: float = Field(
        default=0.33,
        ge=0.0,
        le=1.0,
        description="consensus 진입 임계값. 0.33 = 3개 중 1개 이상 합의",
    )

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=730.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> VwapChannelConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if not (self.scale_short < self.scale_mid < self.scale_long):
            msg = (
                f"scale_short ({self.scale_short}) < scale_mid ({self.scale_mid}) "
                f"< scale_long ({self.scale_long}) 필수"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.scale_long, self.vol_window, self.atr_period) + 10
