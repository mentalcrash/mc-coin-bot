"""GBTrend 전략 설정.

모멘텀 중심 12개 feature + GradientBoosting.
CTREND(28 feat) 대비 feature 축소 + 모멘텀 특화로 상관도 차별화.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class GBTrendConfig(BaseModel):
    """GBTrend (Gradient Boosting Momentum Trend) 전략 설정.

    12개 모멘텀 중심 feature를 GradientBoostingRegressor로 결합.
    CTREND와 다른 feature set으로 상관도를 낮춰 앙상블 가치 극대화.

    Features (12):
        - ROC 4종 (5, 10, 21, 63)
        - EMA Cross 2종 (5/20, 10/50)
        - RSI 1종 (14)
        - ADX 1종 (14)
        - ATR Ratio 1종 (14)
        - Vol Ratio 1종 (10/30)
        - Momentum 2종 (5, 21)

    Attributes:
        training_window: Rolling training window (캔들 수).
        prediction_horizon: Forward return prediction 기간.
        n_estimators: GBR tree 수.
        max_depth: 트리 최대 깊이.
        learning_rate: GBR 학습률.
        subsample: Stochastic GBR 샘플 비율.
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
    training_window: int = Field(default=180, ge=60, le=504)
    prediction_horizon: int = Field(default=5, ge=1, le=21)
    n_estimators: int = Field(default=80, ge=10, le=500)
    max_depth: int = Field(default=3, ge=1, le=8)
    learning_rate: float = Field(default=0.05, ge=0.001, le=0.5)
    subsample: float = Field(default=0.8, gt=0.0, le=1.0)

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
    def _validate_cross_fields(self) -> GBTrendConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return self.training_window + 70  # 63-period ROC + buffer
