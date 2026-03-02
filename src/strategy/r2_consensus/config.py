"""R2 Consensus Trend 전략 설정.

선형회귀 R2(추세 일관성)를 3개 스케일(short/mid/long)에서 이진 투표 consensus.
R2 > threshold AND slope 방향 합의 시에만 진입. 교훈 #29 이진 결정 준수.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class R2ConsensusConfig(BaseModel):
    """R2 Consensus Trend 전략 설정.

    Signal Formula:
        1. 3개 스케일(short/mid/long)에서 rolling OLS → slope, R^2 계산
        2. 각 스케일에서 R^2 > r2_threshold이면 vote = sign(slope), 아니면 0
        3. consensus = sum(votes) / 3
        4. |consensus| >= entry_threshold 시 direction = sign(consensus)
        5. strength = direction * vol_scalar * |consensus|

    Attributes:
        lookback_short: 단기 회귀 lookback (bars).
        lookback_mid: 중기 회귀 lookback (bars).
        lookback_long: 장기 회귀 lookback (bars).
        r2_threshold: 각 스케일에서 투표 자격 최소 R^2.
        entry_threshold: consensus 절대값 진입 기준 (0~1).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 12H = 730.0.
        atr_period: ATR 계산 기간 (trailing stop용).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Multi-Scale Regression Parameters ---
    lookback_short: int = Field(
        default=20,
        ge=5,
        le=60,
        description="단기 회귀 lookback (bars).",
    )
    lookback_mid: int = Field(
        default=50,
        ge=20,
        le=120,
        description="중기 회귀 lookback (bars).",
    )
    lookback_long: int = Field(
        default=120,
        ge=40,
        le=300,
        description="장기 회귀 lookback (bars).",
    )
    r2_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=0.9,
        description="각 스케일에서 투표 자격 최소 R^2.",
    )
    entry_threshold: float = Field(
        default=0.34,
        ge=0.0,
        le=1.0,
        description="consensus 절대값 진입 기준 (2/3 이상 합의 = 0.67/3*2 ≈ 0.34).",
    )

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=730.0, gt=0)

    # --- Options ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> Self:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if not (self.lookback_short < self.lookback_mid < self.lookback_long):
            msg = (
                f"Lookbacks must be strictly increasing: "
                f"short({self.lookback_short}) < mid({self.lookback_mid}) < long({self.lookback_long})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.lookback_long, self.vol_window, self.atr_period) + 10
