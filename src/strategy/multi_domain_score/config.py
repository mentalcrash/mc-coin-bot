"""Multi-Domain Score 전략 설정.

4차원(추세/볼륨/파생상품/변동성) soft scoring.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

_WEIGHT_SUM_TOLERANCE = 1e-6


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class MultiDomainScoreConfig(BaseModel):
    """Multi-Domain Score 전략 설정.

    Attributes:
        w_trend: 추세 차원 가중치.
        w_volume: 볼륨 차원 가중치.
        w_derivatives: 파생상품 차원 가중치.
        w_volatility: 변동성 차원 가중치.
        er_window: Efficiency Ratio window.
        sma_window: SMA direction window.
        obv_roc_window: OBV ROC window.
        fr_zscore_window: FR z-score rolling window.
        fr_ma_window: FR MA window.
        fr_score_cap: |fr_z| 정규화 상한.
        rv_short_window: 단기 변동성 window.
        rv_long_window: 장기 변동성 window.
        entry_threshold: 복합 점수 진입 임계값.
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 4H TF 연환산 계수.
        atr_period: ATR 계산 기간.
        short_mode: 숏 포지션 허용 모드.
    """

    model_config = ConfigDict(frozen=True)

    # --- Dimension Weights ---
    w_trend: float = Field(default=0.35, ge=0.0, le=1.0)
    w_volume: float = Field(default=0.25, ge=0.0, le=1.0)
    w_derivatives: float = Field(default=0.25, ge=0.0, le=1.0)
    w_volatility: float = Field(default=0.15, ge=0.0, le=1.0)

    # --- Per-Dimension Parameters ---
    er_window: int = Field(default=30, ge=5, le=120)
    sma_window: int = Field(default=30, ge=5, le=120)
    obv_roc_window: int = Field(default=10, ge=3, le=30)
    fr_zscore_window: int = Field(default=42, ge=20, le=200)
    fr_ma_window: int = Field(default=9, ge=3, le=30)
    fr_score_cap: float = Field(default=3.0, ge=1.0, le=5.0)
    rv_short_window: int = Field(default=6, ge=3, le=20)
    rv_long_window: int = Field(default=42, ge=20, le=120)

    # --- Composite ---
    entry_threshold: float = Field(default=0.45, ge=0.2, le=0.8)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=120)
    min_volatility: float = Field(default=0.05, gt=0.0, le=0.20)
    annualization_factor: float = Field(default=2190.0, gt=0.0)

    # --- ATR ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> MultiDomainScoreConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        weight_sum = self.w_trend + self.w_volume + self.w_derivatives + self.w_volatility
        if abs(weight_sum - 1.0) > _WEIGHT_SUM_TOLERANCE:
            msg = f"Dimension weights must sum to 1.0, got {weight_sum}"
            raise ValueError(msg)
        if self.rv_short_window >= self.rv_long_window:
            msg = (
                f"rv_short_window ({self.rv_short_window}) "
                f"must be < rv_long_window ({self.rv_long_window})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            max(
                self.er_window,
                self.sma_window,
                self.fr_zscore_window,
                self.rv_long_window,
                self.vol_window,
            )
            + 10
        )
