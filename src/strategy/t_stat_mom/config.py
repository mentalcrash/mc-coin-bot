"""T-Stat Momentum 전략 설정.

수익률 t-statistic 기반 통계적 유의성 모멘텀.
Multi-lookback t-stat blend + tanh 연속 포지션.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class TStatMomConfig(BaseModel):
    """T-Stat Momentum 전략 설정.

    Attributes:
        fast_lookback: 단기 t-stat 계산 lookback (bars).
        mid_lookback: 중기 t-stat 계산 lookback (bars).
        slow_lookback: 장기 t-stat 계산 lookback (bars).
        entry_threshold: t-stat blend가 이 값을 초과하면 진입.
        exit_threshold: t-stat blend가 이 값 미만이면 청산 (양방향).
        tanh_scale: tanh 스케일링 계수 (strength 연속화).
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
    fast_lookback: int = Field(default=20, ge=5, le=60)
    mid_lookback: int = Field(default=40, ge=15, le=120)
    slow_lookback: int = Field(default=80, ge=30, le=200)
    entry_threshold: float = Field(default=1.0, ge=0.3, le=3.0)
    exit_threshold: float = Field(default=0.5, ge=0.0, le=2.0)
    tanh_scale: float = Field(default=0.5, gt=0.0, le=2.0)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=730.0, gt=0.0)

    # --- Options ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> TStatMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.entry_threshold < self.exit_threshold:
            msg = (
                f"entry_threshold ({self.entry_threshold}) "
                f"must be >= exit_threshold ({self.exit_threshold})"
            )
            raise ValueError(msg)
        if not (self.fast_lookback < self.mid_lookback < self.slow_lookback):
            msg = (
                f"Lookbacks must be strictly increasing: "
                f"fast({self.fast_lookback}) < mid({self.mid_lookback}) < slow({self.slow_lookback})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.slow_lookback, self.vol_window) + 10
