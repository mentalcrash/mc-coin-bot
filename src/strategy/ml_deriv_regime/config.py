"""ML Derivatives Regime 전략 설정.

파생상품 포지셔닝 데이터(FR)에 ML Elastic Net을 적용하여
기술지표와 독립적인 derivatives-only alpha를 추구.
레짐 확률을 ML input feature로 사용.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class MlDerivRegimeConfig(BaseModel):
    """ML Derivatives Regime 전략 설정.

    Derivatives 데이터(funding rate)에서 ML feature를 추출하고
    Rolling Elastic Net으로 forward return을 예측.
    레짐 확률(p_trending, p_ranging, p_volatile)을 feature로 활용.

    Attributes:
        training_window: Rolling training window (캔들 수)
        prediction_horizon: Forward return prediction 기간
        alpha: Elastic Net L1 ratio (0=Ridge, 1=Lasso)
        fr_lookback_short: 단기 FR 평균 window
        fr_lookback_long: 장기 FR 평균 window
        fr_zscore_window: FR Z-score window
        vol_target: 연환산 변동성 타겟
        vol_window: 변동성 계산 rolling window
        min_volatility: 변동성 하한
        annualization_factor: TF별 연환산 계수
        short_mode: 숏 포지션 허용 모드
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율
    """

    model_config = ConfigDict(frozen=True)

    # --- ML Parameters ---
    training_window: int = Field(default=252, ge=60, le=504)
    prediction_horizon: int = Field(default=5, ge=1, le=21)
    alpha: float = Field(default=0.5, ge=0.01, le=1.0)

    # --- Derivatives Feature Parameters ---
    fr_lookback_short: int = Field(default=3, ge=1, le=30)
    fr_lookback_long: int = Field(default=21, ge=5, le=90)
    fr_zscore_window: int = Field(default=63, ge=10, le=252)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=365.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> Self:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.fr_lookback_short >= self.fr_lookback_long:
            msg = (
                f"fr_lookback_short ({self.fr_lookback_short}) "
                f"must be < fr_lookback_long ({self.fr_lookback_long})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return self.training_window + 50
