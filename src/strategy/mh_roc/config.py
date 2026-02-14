"""Multi-Horizon ROC Ensemble 전략 설정.

다중 룩백(6/18/42/90 bars) ROC의 부호 투표(voting)로 robust한 추세 신호 생성.
교훈 #1: 앙상블 원리. 비선형 결합으로 conviction scalar 천장 회피.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class MhRocConfig(BaseModel):
    """Multi-Horizon ROC Ensemble 전략 설정.

    Attributes:
        roc_short: 단기 ROC lookback (bars).
        roc_medium_short: 중단기 ROC lookback (bars).
        roc_medium_long: 중장기 ROC lookback (bars).
        roc_long: 장기 ROC lookback (bars).
        vote_threshold: 롱/숏 진입에 필요한 최소 투표 수 (1~4).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 4H TF 연환산 계수 (2190).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Multi-Horizon ROC Lookbacks ---
    roc_short: int = Field(default=6, ge=2, le=50)
    roc_medium_short: int = Field(default=18, ge=5, le=100)
    roc_medium_long: int = Field(default=42, ge=10, le=200)
    roc_long: int = Field(default=90, ge=20, le=365)
    vote_threshold: int = Field(default=3, ge=1, le=4)

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
    def _validate_cross_fields(self) -> MhRocConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if not (self.roc_short < self.roc_medium_short < self.roc_medium_long < self.roc_long):
            msg = (
                "ROC lookbacks must be strictly increasing: "
                f"{self.roc_short} < {self.roc_medium_short} < "
                f"{self.roc_medium_long} < {self.roc_long}"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.roc_long, self.vol_window) + 10
