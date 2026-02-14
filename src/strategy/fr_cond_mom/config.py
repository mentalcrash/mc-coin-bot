"""FR Conditional Momentum 전략 설정.

FR z-score 극단치=과밀 포지셔닝 시 모멘텀 conviction 조건부 조절.
BIS WP 1087 + Inan 2025 실증.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class FrCondMomConfig(BaseModel):
    """FR Conditional Momentum 전략 설정.

    Signal Formula:
        1. mom_signal = sign(close.pct_change(mom_lookback))
        2. fr_zscore = rolling_zscore(funding_rate_ma, fr_zscore_window)
        3. conviction = 1.0 if |fr_zscore| < fr_neutral_zone (정상)
                       = fr_dampening if |fr_zscore| >= fr_extreme_threshold (극단)
                       = linear interpolation in between
        4. Long: mom_signal > 0, strength = vol_scalar * conviction
        5. Short (HEDGE_ONLY): mom_signal < 0 + drawdown 조건

    Attributes:
        mom_lookback: 모멘텀 방향 결정 lookback (6H bar 수).
        mom_ma_window: 모멘텀 시그널 smoothing MA window.
        fr_ma_window: Funding rate 이동평균 window.
        fr_zscore_window: FR z-score 정규화 rolling window.
        fr_neutral_zone: FR z-score 중립 구간 (conviction 미감쇄).
        fr_extreme_threshold: FR z-score 극단 임계값 (conviction 최대 감쇄).
        fr_dampening: 극단 FR 시 conviction 감쇄 비율 (0~1).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 6H TF 연환산 계수 (1460).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Momentum Parameters ---
    mom_lookback: int = Field(default=20, ge=5, le=120)
    mom_ma_window: int = Field(default=5, ge=2, le=30)

    # --- Funding Rate Parameters ---
    fr_ma_window: int = Field(default=9, ge=3, le=60)
    fr_zscore_window: int = Field(default=60, ge=20, le=200)
    fr_neutral_zone: float = Field(default=0.5, ge=0.0, le=2.0)
    fr_extreme_threshold: float = Field(default=2.0, ge=0.5, le=5.0)
    fr_dampening: float = Field(default=0.3, ge=0.0, le=1.0)

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
    def _validate_cross_fields(self) -> FrCondMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.fr_neutral_zone >= self.fr_extreme_threshold:
            msg = (
                f"fr_neutral_zone ({self.fr_neutral_zone}) "
                f"must be < fr_extreme_threshold ({self.fr_extreme_threshold})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            max(self.mom_lookback + self.mom_ma_window, self.fr_zscore_window, self.vol_window) + 10
        )
