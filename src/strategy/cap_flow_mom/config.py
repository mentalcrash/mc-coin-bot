"""Capital Flow Momentum 전략 설정.

12H 듀얼스피드 ROC 모멘텀 + 1D Stablecoin supply ROC 확신도 가중.
자본 유입/유출 방향과 가격 모멘텀 정렬 시 확신도 강화, 괴리 시 감쇄.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class CapFlowMomConfig(BaseModel):
    """Capital Flow Momentum 전략 설정.

    Attributes:
        fast_roc_period: 빠른 ROC 기간 (12H bars)
        slow_roc_period: 느린 ROC 기간 (12H bars)
        roc_threshold: ROC 시그널 활성화 임계값
        stablecoin_roc_window: Stablecoin supply ROC 계산 윈도우 (1D bars)
        stablecoin_boost: 모멘텀/자본흐름 정렬 시 확신도 부스트 계수
        stablecoin_dampen: 모멘텀/자본흐름 괴리 시 확신도 감쇄 계수
        vol_target: 연환산 변동성 타겟 (0~1)
        vol_window: 변동성 계산 rolling window
        min_volatility: 변동성 하한 (0 나눗셈 방지)
        annualization_factor: TF별 연환산 계수 (12H = 730)
        short_mode: 숏 포지션 허용 모드
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0)
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    fast_roc_period: int = Field(default=6, ge=2, le=30)
    slow_roc_period: int = Field(default=30, ge=10, le=120)
    roc_threshold: float = Field(default=0.01, ge=0.001, le=0.10)
    stablecoin_roc_window: int = Field(default=14, ge=5, le=60)
    stablecoin_boost: float = Field(default=1.3, ge=1.0, le=2.0)
    stablecoin_dampen: float = Field(default=0.5, ge=0.1, le=1.0)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=730.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> CapFlowMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.fast_roc_period >= self.slow_roc_period:
            msg = (
                f"fast_roc_period ({self.fast_roc_period}) must be < "
                f"slow_roc_period ({self.slow_roc_period})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.slow_roc_period, self.vol_window) + 10
