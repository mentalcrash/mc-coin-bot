"""Exchange Flow Momentum 전략 설정.

거래소 순유출(outflow>inflow) = 축적 = 공급 감소 = 상승 압력.
Flow 변화율(momentum)이 가격 선행. BTC/ETH 전용.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class ExFlowMomConfig(BaseModel):
    """Exchange Flow Momentum 전략 설정.

    Attributes:
        flow_window: 거래소 순유출 smoothing 기간.
        flow_mom_window: 순유출 변화율 계산 기간.
        flow_threshold: 시그널 발생 임계값 (z-score).
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 1D TF 연환산 계수.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준.
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    flow_window: int = Field(default=14, ge=3, le=60)
    flow_mom_window: int = Field(default=7, ge=1, le=30)
    flow_threshold: float = Field(default=0.5, ge=0.0, le=3.0)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=365.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> ExFlowMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.flow_window + self.flow_mom_window, self.vol_window) + 10
