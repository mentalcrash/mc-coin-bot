"""Donchian Multi-Scale 전략 설정.

3-scale Donchian breakout 앙상블(20/40/80). Multi-lookback 합의로 regime-robust.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class DonchMultiConfig(BaseModel):
    """Donchian Multi-Scale 전략 설정.

    3개 Donchian Channel(20/40/80) breakout 시그널을 앙상블하여
    추세 방향 합의(consensus)를 도출한다.

    Signal Logic:
        1. 각 lookback에 대해 Donchian Channel 계산
        2. signal_i = +1 if close > prev_upper, -1 if close < prev_lower, 0 hold
        3. consensus = mean(signal_1, signal_2, signal_3)
        4. direction = sign(consensus) if |consensus| >= entry_threshold else 0
        5. strength = |consensus| * vol_scalar

    Attributes:
        lookback_short: 단기 Donchian lookback (bars).
        lookback_mid: 중기 Donchian lookback (bars).
        lookback_long: 장기 Donchian lookback (bars).
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
    lookback_short: int = Field(default=20, ge=5, le=100)
    lookback_mid: int = Field(default=40, ge=10, le=200)
    lookback_long: int = Field(default=80, ge=20, le=400)
    entry_threshold: float = Field(
        default=0.34,
        ge=0.0,
        le=1.0,
        description="consensus 진입 임계값. 0.34 = 3개 중 2개 합의 필요",
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
    def _validate_cross_fields(self) -> DonchMultiConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if not (self.lookback_short < self.lookback_mid < self.lookback_long):
            msg = (
                f"lookback_short ({self.lookback_short}) < lookback_mid ({self.lookback_mid}) "
                f"< lookback_long ({self.lookback_long}) 필수"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.lookback_long, self.vol_window) + 10
