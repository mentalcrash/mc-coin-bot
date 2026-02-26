"""Donchian Cascade MTF 전략 설정.

12H-equivalent Donchian channel을 4H 해상도로 계산하고,
4H EMA confirmation으로 진입 타이밍을 최적화한다.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class DonchCascadeConfig(BaseModel):
    """Donchian Cascade MTF 전략 설정.

    HTF (12H-equivalent) Donchian consensus로 방향을 결정하고,
    LTF (4H) EMA confirmation으로 진입 타이밍을 최적화한다.

    Signal Logic:
        1. 각 lookback * htf_multiplier에 대해 4H Donchian Channel 계산
        2. 3-scale consensus (donch-multi 동일)
        3. |consensus| >= entry_threshold → htf_direction 결정
        4. 4H EMA confirmation: close > EMA(confirm_period) → 진입 허용
        5. max_wait_bars 초과 시 강제 진입
        6. strength = |consensus| * vol_scalar

    Attributes:
        lookback_short: HTF 단기 Donchian lookback (12H 스케일 bars).
        lookback_mid: HTF 중기 Donchian lookback (12H 스케일 bars).
        lookback_long: HTF 장기 Donchian lookback (12H 스케일 bars).
        entry_threshold: consensus 진입 임계값 (0~1).
        htf_multiplier: HTF/LTF 비율. 12H/4H=3.
        confirm_ema_period: 4H EMA 기간 (진입 확인용).
        max_wait_bars: 확인 대기 최대 4H bar 수.
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window (4H bars).
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 4H TF 연환산 계수 (2190 = 6*365).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- HTF Donchian (12H-equivalent base lookbacks) ---
    lookback_short: int = Field(default=20, ge=5, le=100)
    lookback_mid: int = Field(default=40, ge=10, le=200)
    lookback_long: int = Field(default=80, ge=20, le=400)
    entry_threshold: float = Field(
        default=0.34,
        ge=0.0,
        le=1.0,
        description="consensus 진입 임계값. 0.34 = 3개 중 2개 합의 필요",
    )
    htf_multiplier: int = Field(
        default=3,
        ge=2,
        le=6,
        description="HTF/LTF 비율. 12H/4H=3, 1D/4H=6",
    )

    # --- LTF Entry Confirmation ---
    confirm_ema_period: int = Field(
        default=5,
        ge=2,
        le=30,
        description="4H EMA 기간. 모멘텀 확인용 단기 EMA",
    )
    max_wait_bars: int = Field(
        default=3,
        ge=1,
        le=10,
        description="확인 대기 최대 bar 수. 3 = 1 HTF period (12H)",
    )

    # --- Vol-Target Parameters (4H scale) ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=90, ge=10, le=500)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=2190.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> DonchCascadeConfig:
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

    def actual_lookbacks(self) -> tuple[int, int, int]:
        """실제 Donchian 계산에 사용할 lookback (base x htf_multiplier)."""
        m = self.htf_multiplier
        return (
            self.lookback_short * m,
            self.lookback_mid * m,
            self.lookback_long * m,
        )

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        max_lookback = self.lookback_long * self.htf_multiplier
        return max(max_lookback, self.vol_window) + 10
