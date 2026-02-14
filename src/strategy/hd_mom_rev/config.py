"""Half-Day Momentum-Reversal 전략 설정.

12H 전반부 수익률로 후반부 방향 예측.
정상일(낮은 vol) momentum, 급변일(높은 vol) reversal.
Wen et al. 2022 실증.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class HdMomRevConfig(BaseModel):
    """Half-Day Momentum-Reversal 전략 설정.

    Signal Formula:
        1. half_return = log(close / open) per 12H bar (intrabar 전반부 proxy)
        2. jump_score = |half_return| / realized_vol (정규화된 jump 크기)
        3. If jump_score < jump_threshold -> momentum: direction = sign(half_return)
        4. If jump_score >= jump_threshold -> reversal: direction = -sign(half_return)
        5. strength = direction * vol_scalar * confidence
        6. confidence = min(jump_score / jump_threshold, 1) (momentum)
                      = min((jump_score - jump_threshold) / jump_threshold, 1) (reversal)

    Attributes:
        jump_threshold: 정상/급변 구분 임계값 (정규화된 return 크기).
        half_return_ma: half_return smoothing MA window.
        confidence_cap: confidence 최대값 제한.
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 12H TF 연환산 계수 (730).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    jump_threshold: float = Field(default=2.0, ge=0.5, le=5.0)
    half_return_ma: int = Field(default=3, ge=1, le=20)
    confidence_cap: float = Field(default=1.0, gt=0.0, le=2.0)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=730.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> HdMomRevConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.half_return_ma, self.vol_window) + 10
