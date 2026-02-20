"""EMA Ribbon Momentum 설정 모델."""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class EmaRibbonMomConfig(BaseModel):
    """피보나치 EMA 리본 + ROC 모멘텀 설정.

    5개 피보나치 EMA(8,13,21,34,55)의 정렬도(alignment)와
    ROC 모멘텀 확인으로 추세 성숙도를 측정.
    """

    model_config = ConfigDict(frozen=True)

    # 피보나치 EMA 기간 (5개)
    ema_periods: tuple[int, ...] = Field(default=(8, 13, 21, 34, 55))

    # ROC 모멘텀 기간
    roc_period: int = Field(default=21, ge=5, le=63)

    # Alignment 임계값 (0~1, 이상이면 진입)
    alignment_threshold: float = Field(default=0.7, ge=0.3, le=1.0)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=252)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=365.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    _MIN_EMA_COUNT: int = 3

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> EmaRibbonMomConfig:
        """교차 필드 검증: EMA 순서 + vol_target >= min_volatility."""
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        periods = self.ema_periods
        if len(periods) < self._MIN_EMA_COUNT:
            msg = f"ema_periods must have at least 3 elements, got {len(periods)}"
            raise ValueError(msg)
        for i in range(1, len(periods)):
            if periods[i] <= periods[i - 1]:
                msg = f"ema_periods must be strictly ascending: {periods}"
                raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        max_ema = max(self.ema_periods)
        return max_ema + self.vol_window + 10
