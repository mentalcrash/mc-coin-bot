"""Fragility-Aware Momentum 전략 설정.

VoV가 regime 전환 감지기. Low VoV=안정(추세 신뢰), High VoV=전환(축소).
GK vol percentile로 추세 확인도 조절. 과신→취약성 축적의 행동편향 활용.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class FragilityMomConfig(BaseModel):
    """Fragility-Aware Momentum 전략 설정.

    Signal Logic:
        1. VoV percentile < threshold → stable regime → 모멘텀 신뢰
        2. GK vol percentile → 변동성 수준에 따른 확신도 가중
           - Low GK pct: 시장 잠잠 → 추세 지속 확신 높음
           - High GK pct: 시장 활발 → 확신 감소 (과열 = 취약)
        3. conviction = (1 - gk_vol_pct) * (1 - vov_pct)
           → 낮은 VoV + 낮은 GK vol일수록 높은 확신
        4. strength = direction * vol_scalar * conviction

    Attributes:
        gk_window: GK volatility rolling window (bars).
        vov_window: VoV (vol-of-vol) rolling window (bars).
        vov_percentile_window: VoV percentile rank 계산 window.
        vov_threshold: VoV percentile 임계값 (이하일 때만 진입).
        gk_vol_percentile_window: GK vol percentile rank 계산 window.
        mom_lookback: 가격 모멘텀 lookback (bars).
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
    gk_window: int = Field(default=20, ge=5, le=100)
    vov_window: int = Field(default=20, ge=5, le=100)
    vov_percentile_window: int = Field(default=252, ge=30, le=500)
    vov_threshold: float = Field(default=0.35, gt=0.0, le=1.0)
    gk_vol_percentile_window: int = Field(default=252, ge=30, le=500)
    mom_lookback: int = Field(default=42, ge=3, le=100)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=730.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> FragilityMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            max(
                self.gk_window + self.vov_window,
                self.vov_percentile_window,
                self.gk_vol_percentile_window,
                self.mom_lookback,
                self.vol_window,
            )
            + 10
        )
