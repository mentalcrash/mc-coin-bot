"""Directional-Asymmetric Multi-Scale Momentum 전략 설정.

핵심 혁신: UP/DOWN 모멘텀에 서로 다른 lookback window를 적용.
상승은 느리게 확인(긴 lookback), 하락은 빠르게 반응(짧은 lookback).
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class AsymmetricTrend8hConfig(BaseModel):
    """Directional-Asymmetric Multi-Scale Momentum 전략 설정.

    UP 모멘텀은 긴 lookback으로 추세 확인(느린 진입),
    DOWN 모멘텀은 짧은 lookback으로 빠른 반응(빠른 진입/손절).
    3-scale consensus 투표로 노이즈 필터링.

    Signal Logic:
        1. UP ROC: 3-scale(15/30/63) sign 평균 → up_score
        2. DN ROC: 3-scale(3/6/15) sign 평균 → dn_score
        3. up_score >= consensus_threshold → long
        4. dn_score <= -consensus_threshold → short
        5. strength = direction * relevant_score_magnitude * vol_scalar

    Attributes:
        up_lookback_short: Up-momentum 단기 lookback (8H bars, ~5일).
        up_lookback_mid: Up-momentum 중기 lookback (8H bars, ~10일).
        up_lookback_long: Up-momentum 장기 lookback (8H bars, ~21일).
        dn_lookback_short: Down-momentum 단기 lookback (8H bars, ~1일).
        dn_lookback_mid: Down-momentum 중기 lookback (8H bars, ~2일).
        dn_lookback_long: Down-momentum 장기 lookback (8H bars, ~5일).
        consensus_threshold: 진입 최소 consensus 비율 (0.67 = 2/3 스케일 합의).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 8H TF 연환산 계수 (3 bars/day x 365).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Up-Momentum Lookbacks (slow confirmation) ---
    up_lookback_short: int = Field(
        default=15,
        ge=5,
        le=50,
        description="Up-momentum short lookback (8H bars, ~5 days)",
    )
    up_lookback_mid: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Up-momentum mid lookback (8H bars, ~10 days)",
    )
    up_lookback_long: int = Field(
        default=63,
        ge=20,
        le=200,
        description="Up-momentum long lookback (8H bars, ~21 days)",
    )

    # --- Down-Momentum Lookbacks (fast reaction) ---
    dn_lookback_short: int = Field(
        default=3,
        ge=1,
        le=15,
        description="Down-momentum short lookback (8H bars, ~1 day)",
    )
    dn_lookback_mid: int = Field(
        default=6,
        ge=2,
        le=30,
        description="Down-momentum mid lookback (8H bars, ~2 days)",
    )
    dn_lookback_long: int = Field(
        default=15,
        ge=5,
        le=50,
        description="Down-momentum long lookback (8H bars, ~5 days)",
    )

    # --- Consensus ---
    consensus_threshold: float = Field(
        default=0.67,
        ge=0.33,
        le=1.0,
        description="Min consensus ratio for entry (0.67 = 2/3 scales agree)",
    )

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=1095.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> AsymmetricTrend8hConfig:
        if not (self.up_lookback_short < self.up_lookback_mid < self.up_lookback_long):
            msg = (
                f"up_lookback_short ({self.up_lookback_short}) < up_lookback_mid ({self.up_lookback_mid}) "
                f"< up_lookback_long ({self.up_lookback_long}) 필수"
            )
            raise ValueError(msg)
        if not (self.dn_lookback_short < self.dn_lookback_mid < self.dn_lookback_long):
            msg = (
                f"dn_lookback_short ({self.dn_lookback_short}) < dn_lookback_mid ({self.dn_lookback_mid}) "
                f"< dn_lookback_long ({self.dn_lookback_long}) 필수"
            )
            raise ValueError(msg)
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.up_lookback_long, self.vol_window) + 10
