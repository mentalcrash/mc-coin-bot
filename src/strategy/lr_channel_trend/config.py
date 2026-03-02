"""LR-Channel Multi-Scale Trend 전략 설정.

Rolling OLS 선형회귀 채널 x 3스케일(20/60/150) consensus breakout.
채널 = regression_line +/- k * residual_std. 가격을 명시적으로 detrend하여
추세 주변의 노이즈만 측정한다.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class LrChannelTrendConfig(BaseModel):
    """LR-Channel Multi-Scale Trend 전략 설정.

    Rolling OLS 선형회귀 채널 x 3스케일(20/60/150) consensus breakout.
    3개 sub-signal의 평균(consensus) -> sign -> vol scaling.

    Signal Logic:
        1. 각 스케일에 대해 LR channel breakout 시그널 계산 (총 3개)
        2. consensus = mean(signal_1, signal_2, signal_3)
        3. direction = sign(consensus) if |consensus| >= entry_threshold else 0
        4. strength = |consensus| * vol_scalar

    Attributes:
        scale_short: 단기 스케일 (bars).
        scale_mid: 중기 스케일 (bars).
        scale_long: 장기 스케일 (bars).
        channel_multiplier: 잔차 표준편차 배수 (채널 폭 결정).
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
    channel_multiplier: float = Field(
        default=2.0,
        ge=0.5,
        le=5.0,
        description="잔차 표준편차 배수. 채널 폭 결정",
    )
    entry_threshold: float = Field(
        default=0.22,
        ge=0.0,
        le=1.0,
        description="consensus 진입 임계값. 0.22 = 3개 중 ~1개 합의로 진입",
    )

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=730.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.DISABLED)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> LrChannelTrendConfig:
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
        return max(self.scale_long, self.vol_window) + 10
