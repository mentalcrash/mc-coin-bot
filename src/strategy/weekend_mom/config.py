"""Weekend-Momentum 전략 설정.

주말 returns에 가중치를 부여한 multi-scale momentum.
기관 부재 시 retail behavioral momentum persistence 포착.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class WeekendMomConfig(BaseModel):
    """Weekend-Momentum 전략 설정.

    Attributes:
        fast_lookback: Fast momentum lookback (bars).
        slow_lookback: Slow momentum lookback (bars).
        weekend_boost: Weekend return 가중치 배수 (1.0 = 균등).
        mom_threshold: Momentum 시그널 진입 임계값.
        short_mom_threshold: Short 진입 momentum 임계값 (절대값).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 12H TF 연환산 계수 (730).
        atr_period: ATR 계산 기간.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    fast_lookback: int = Field(default=20, ge=5, le=100)
    slow_lookback: int = Field(default=60, ge=20, le=200)
    weekend_boost: float = Field(default=2.0, ge=1.0, le=5.0)
    mom_threshold: float = Field(default=0.0, ge=0.0, le=0.5)
    short_mom_threshold: float = Field(default=0.0, ge=0.0, le=0.5)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=730.0, gt=0.0)

    # --- Options ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> WeekendMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.fast_lookback >= self.slow_lookback:
            msg = (
                f"fast_lookback ({self.fast_lookback}) must be < "
                f"slow_lookback ({self.slow_lookback})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수.

        mom_score는 slow_lookback rolling sum + slow_lookback rolling z-score = 2 * slow_lookback.
        """
        # mom_score = rolling_sum(slow) + rolling_zscore(slow) = 2 * slow_lookback
        mom_score_warmup = self.slow_lookback * 2
        return max(mom_score_warmup, self.vol_window, self.atr_period) + 10
