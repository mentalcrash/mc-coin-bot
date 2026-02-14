"""Variance Decomposition Momentum 전략 설정.

Realized variance를 good(상방)과 bad(하방) semivariance로 분해하여
모멘텀 품질을 측정. good_var 지배적 추세는 지속, bad_var 지배적 추세는 붕괴.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class VardecompMomConfig(BaseModel):
    """Variance Decomposition Momentum 전략 설정.

    Attributes:
        semivar_window: Semivariance 계산 rolling window (bars).
        mom_lookback: 가격 모멘텀 계산 lookback (bars).
        var_ratio_threshold: good_var/(good_var+bad_var) 진입 임계값.
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 1D TF 연환산 계수 (365).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    semivar_window: int = Field(default=30, ge=5, le=200)
    mom_lookback: int = Field(default=20, ge=3, le=100)
    var_ratio_threshold: float = Field(default=0.55, ge=0.5, le=0.95)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=365.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> VardecompMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.semivar_window, self.mom_lookback, self.vol_window) + 10
