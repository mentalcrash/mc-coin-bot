"""Funding Rate Event Trigger + 12H Momentum Context 전략 설정 (8H TF).

펀딩비 극단 z-score 이벤트와 EMA 추세 컨텍스트를 결합하여
crowded positioning reversal을 포착한다.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class FrEventMtf8hConfig(BaseModel):
    """Funding Rate Event Trigger + 12H Momentum Context 전략 설정.

    8H TF에서 펀딩비 z-score 극단값(crowded positioning)이 EMA 추세와
    반대 방향일 때 mean-reversion 이벤트를 포착.

    Signal Logic:
        1. FR z-score < -threshold & EMA bull → long (shorts crowded + uptrend)
        2. FR z-score > +threshold & EMA bear → short (longs crowded + downtrend)
        3. min_hold_bars 동안 포지션 유지 (whipsaw 방지)
        4. strength = |direction| * vol_scalar * clipped_fr_z

    Attributes:
        fr_ma_window: FR 이동평균 윈도우 (8H bars).
        fr_zscore_window: FR z-score lookback (8H bars, ~30 days).
        fr_extreme_threshold: FR z-score 극단 임계값.
        ema_fast_period: 빠른 EMA 기간 (8H bars, ~5일).
        ema_slow_period: 느린 EMA 기간 (8H bars, ~15일).
        min_hold_bars: 최소 보유 기간 (8H bars = 24H).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 8H TF 연환산 계수 (1095 = 3 bars/day x 365).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    fr_ma_window: int = Field(
        default=3, ge=1, le=10, description="FR moving average window (8H bars)"
    )
    fr_zscore_window: int = Field(
        default=90,
        ge=30,
        le=360,
        description="FR z-score lookback (8H bars, ~30 days)",
    )
    fr_extreme_threshold: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="FR z-score extreme threshold",
    )
    ema_fast_period: int = Field(
        default=15,
        ge=5,
        le=50,
        description="Fast EMA on 8H bars (~5 days, approximates 12H 10-bar)",
    )
    ema_slow_period: int = Field(
        default=45,
        ge=10,
        le=150,
        description="Slow EMA on 8H bars (~15 days, approximates 12H 30-bar)",
    )
    min_hold_bars: int = Field(
        default=3,
        ge=1,
        le=12,
        description="Minimum holding period (8H bars = 24H)",
    )

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=1095.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> FrEventMtf8hConfig:
        if self.ema_fast_period >= self.ema_slow_period:
            msg = (
                f"ema_fast_period ({self.ema_fast_period}) must be < "
                f"ema_slow_period ({self.ema_slow_period})"
            )
            raise ValueError(msg)
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.fr_zscore_window, self.ema_slow_period, self.vol_window) + 10
