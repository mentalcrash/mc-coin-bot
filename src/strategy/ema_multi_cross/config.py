"""EMA Multi-Cross 설정 모델."""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class EmaMultiCrossConfig(BaseModel):
    """3쌍 EMA 크로스 합의 투표 설정.

    3개 EMA pair (단기/중기/장기)의 방향 투표로 시그널 생성.
    2/3 이상 합의 시에만 진입.
    """

    model_config = ConfigDict(frozen=True)

    # 단기 EMA pair
    pair1_fast: int = Field(default=8, ge=3, le=30)
    pair1_slow: int = Field(default=21, ge=10, le=60)

    # 중기 EMA pair
    pair2_fast: int = Field(default=20, ge=10, le=60)
    pair2_slow: int = Field(default=50, ge=30, le=120)

    # 장기 EMA pair
    pair3_fast: int = Field(default=50, ge=20, le=120)
    pair3_slow: int = Field(default=100, ge=60, le=300)

    # 합의 임계값 (2 = 2/3 합의, 3 = 만장일치)
    min_votes: int = Field(default=2, ge=2, le=3)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=252)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=365.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> EmaMultiCrossConfig:
        """교차 필드 검증: pair 순서 + vol_target >= min_volatility."""
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.pair1_fast >= self.pair1_slow:
            msg = f"pair1: fast({self.pair1_fast}) must be < slow({self.pair1_slow})"
            raise ValueError(msg)
        if self.pair2_fast >= self.pair2_slow:
            msg = f"pair2: fast({self.pair2_fast}) must be < slow({self.pair2_slow})"
            raise ValueError(msg)
        if self.pair3_fast >= self.pair3_slow:
            msg = f"pair3: fast({self.pair3_fast}) must be < slow({self.pair3_slow})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        max_period = max(self.pair1_slow, self.pair2_slow, self.pair3_slow)
        return max_period + self.vol_window + 10
