"""CTREND-X 전략 설정.

CTREND의 확장판: GradientBoosting으로 비선형 패턴 캡처.
동일 28개 feature + GBR(Gradient Boosting Regressor).
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class CTRENDXConfig(BaseModel):
    """CTREND-X (Gradient Boosting Trend) 전략 설정.

    CTREND와 동일한 28개 기술적 지표를 Rolling GradientBoostingRegressor로
    결합하여 forward return을 예측합니다.
    ElasticNet 대비 비선형 상호작용(interaction)을 캡처합니다.

    Attributes:
        training_window: Rolling training window (캔들 수).
        prediction_horizon: Forward return prediction 기간.
        n_estimators: GBR tree 수.
        max_depth: 트리 최대 깊이 (과적합 제어).
        learning_rate: GBR 학습률.
        subsample: 각 트리의 샘플 비율 (stochastic GBR).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 1D TF 연환산 계수 (365).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- ML Parameters ---
    training_window: int = Field(
        default=252,
        ge=60,
        le=504,
        description="Rolling training window (캔들 수).",
    )
    prediction_horizon: int = Field(
        default=5,
        ge=1,
        le=21,
        description="Forward return prediction horizon.",
    )
    n_estimators: int = Field(
        default=100,
        ge=10,
        le=500,
        description="GBR tree 수.",
    )
    max_depth: int = Field(
        default=3,
        ge=1,
        le=8,
        description="트리 최대 깊이.",
    )
    learning_rate: float = Field(
        default=0.05,
        ge=0.001,
        le=0.5,
        description="GBR 학습률.",
    )
    subsample: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        description="각 트리의 샘플 비율.",
    )

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
    def _validate_cross_fields(self) -> CTRENDXConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return self.training_window + 50
