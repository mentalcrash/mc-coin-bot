"""Funding Divergence Momentum 전략 설정.

가격 모멘텀과 funding rate 추세의 divergence로 추세 지속 가능성 예측.
FR 하락+가격 상승=유기적 수요(지속), FR 급등+가격 상승=투기과열(청산위험).
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class FundDivMomConfig(BaseModel):
    """Funding Divergence Momentum 전략 설정.

    Attributes:
        mom_lookback: 가격 모멘텀 계산 lookback (bars).
        fr_lookback: Funding rate rolling mean window (bars).
        fr_zscore_window: FR z-score 정규화 window.
        divergence_threshold: Divergence score 진입 임계값.
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 4H TF 연환산 계수 (2190).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    mom_lookback: int = Field(default=18, ge=3, le=100)
    fr_lookback: int = Field(default=6, ge=1, le=50)
    fr_zscore_window: int = Field(default=90, ge=10, le=365)
    divergence_threshold: float = Field(default=0.5, ge=0.0, le=3.0)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=2190.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> FundDivMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.fr_zscore_window, self.mom_lookback, self.vol_window) + 10
