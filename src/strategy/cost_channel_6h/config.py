"""Cost-Penalized Multi-Scale Channel 전략 설정.

3종 채널(Donchian/Keltner/BB) x 3스케일(15/45/120) 앙상블 breakout + 비용 페널티.
포지션 변경 시 기대 이익이 비용을 초과할 때만 시그널 전환하여 거래 비용을 최소화.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class CostChannel6hConfig(BaseModel):
    """Cost-Penalized Multi-Scale Channel 전략 설정.

    3종 채널(Donchian/Keltner/BB) x 3스케일(15/45/120) 앙상블 breakout + 비용 페널티.
    9개 sub-signal의 평균(consensus) → direction → cost penalty filter → vol scaling.

    Signal Logic:
        1. 각 (채널, 스케일) 조합에 대해 breakout 시그널 계산 (총 9개)
        2. consensus = mean(signal_1, ..., signal_9)
        3. raw_direction = sign(consensus) if |consensus| >= entry_threshold else 0
        4. cost filter: 포지션 변경 시 expected_profit > theta * cost 일 때만 변경
        5. strength = |consensus| * vol_scalar

    Attributes:
        scale_short: 단기 스케일 (6H bars, ~2.5일).
        scale_mid: 중기 스케일 (6H bars, ~7.5일).
        scale_long: 장기 스케일 (6H bars, ~20일).
        bb_std_dev: Bollinger Bands 표준편차 배수.
        keltner_multiplier: Keltner Channels ATR 배수.
        entry_threshold: consensus 진입 임계값 (0~1).
        cost_penalty_theta: 비용 페널티 계수. 클수록 거래 빈도 감소.
        round_trip_cost: 단위 포지션 변경당 왕복 거래 비용.
        atr_profit_window: 기대 이익 추정용 ATR 윈도우.
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 6H TF 연환산 계수 (4 bars/day x 365 = 1460).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    scale_short: int = Field(default=15, ge=5, le=80)
    scale_mid: int = Field(default=45, ge=10, le=200)
    scale_long: int = Field(default=120, ge=20, le=400)
    bb_std_dev: float = Field(default=2.0, ge=0.5, le=4.0)
    keltner_multiplier: float = Field(default=1.5, ge=0.5, le=4.0)
    entry_threshold: float = Field(
        default=0.22,
        ge=0.0,
        le=1.0,
        description="consensus 진입 임계값. 0.22 = 9개 중 ~2개 합의로 진입",
    )

    # --- Cost Penalty Parameters ---
    cost_penalty_theta: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Cost penalty. Higher = fewer trades. Signal change only when expected profit > theta * cost",
    )
    round_trip_cost: float = Field(
        default=0.0008,
        ge=0.0,
        le=0.01,
        description="Round-trip cost per unit position change",
    )
    atr_profit_window: int = Field(
        default=20,
        ge=5,
        le=100,
        description="ATR window for expected profit estimation",
    )

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=1460.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> CostChannel6hConfig:
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
        return max(self.scale_long, self.vol_window, self.atr_profit_window) + 10
