"""Vol-Term ML 전략 설정.

다중 RV term structure feature를 Ridge 회귀로 결합하여
변동성 regime 기반 방향성 시그널 생성.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class VolTermMLConfig(BaseModel):
    """Vol-Term ML 전략 설정.

    Features (10):
        - RV 5종 (5, 10, 20, 40, 60)
        - Vol Ratio 3종 (5/20, 10/40, 20/60)
        - Parkinson Vol 1종
        - GK Vol 1종

    Attributes:
        training_window: Rolling training window (캔들 수).
        prediction_horizon: Forward return prediction 기간.
        ridge_alpha: Ridge regularization strength.
        rv_windows: RV 계산에 사용할 window 목록.
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 1D TF 연환산 계수.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준.
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- ML Parameters ---
    training_window: int = Field(default=252, ge=60, le=504)
    prediction_horizon: int = Field(default=5, ge=1, le=21)
    ridge_alpha: float = Field(default=1.0, ge=0.01, le=100.0)

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
    def _validate_cross_fields(self) -> VolTermMLConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return self.training_window + 70  # 60-period RV + buffer
