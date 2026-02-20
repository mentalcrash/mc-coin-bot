"""Regime-Adaptive Multi-Lookback Momentum 전략 설정.

다중 lookback(20d/60d/120d) 모멘텀을 RegimeService 확률 기반으로 연속 가중 혼합.
Trending -> 빠른 반응, Volatile -> 안정적 장기 추세.
레짐은 방향이 아닌 시간 스케일 선택만 담당.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class RegimeAdaptiveMomConfig(BaseModel):
    """Regime-Adaptive Multi-Lookback Momentum 전략 설정.

    다중 시간 스케일 모멘텀을 레짐 확률로 가중 혼합.
    레짐은 방향이 아닌 lookback 선호도만 결정 (anti-pattern #9 회피).

    Attributes:
        fast_lookback: 빠른 모멘텀 lookback (1D bars).
        mid_lookback: 중간 모멘텀 lookback (1D bars).
        slow_lookback: 느린 모멘텀 lookback (1D bars).
        trending_fast_weight: trending 레짐에서 fast mom 가중치.
        trending_mid_weight: trending 레짐에서 mid mom 가중치.
        trending_slow_weight: trending 레짐에서 slow mom 가중치.
        volatile_fast_weight: volatile 레짐에서 fast mom 가중치.
        volatile_mid_weight: volatile 레짐에서 mid mom 가중치.
        volatile_slow_weight: volatile 레짐에서 slow mom 가중치.
        signal_threshold: blended momentum 절대값 최소 진입 임계값.
        trending_vol_target: trending 레짐에서의 vol_target.
        ranging_vol_target: ranging 레짐에서의 vol_target.
        volatile_vol_target: volatile 레짐에서의 vol_target.
        vol_target: 연환산 변동성 타겟 (fallback, 0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 1D TF 연환산 계수 (365).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Multi-Lookback Momentum Parameters ---
    fast_lookback: int = Field(
        default=20,
        ge=5,
        le=60,
        description="Fast momentum lookback (1D bars).",
    )
    mid_lookback: int = Field(
        default=60,
        ge=20,
        le=120,
        description="Mid momentum lookback (1D bars).",
    )
    slow_lookback: int = Field(
        default=120,
        ge=60,
        le=252,
        description="Slow momentum lookback (1D bars).",
    )

    # --- Trending Regime Weights ---
    trending_fast_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Fast mom weight in trending regime.",
    )
    trending_mid_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Mid mom weight in trending regime.",
    )
    trending_slow_weight: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Slow mom weight in trending regime.",
    )

    # --- Volatile Regime Weights ---
    volatile_fast_weight: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fast mom weight in volatile regime.",
    )
    volatile_mid_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Mid mom weight in volatile regime.",
    )
    volatile_slow_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Slow mom weight in volatile regime.",
    )

    # --- Signal Threshold ---
    signal_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=0.2,
        description="Minimum |blended_momentum| to generate signal.",
    )

    # --- Regime-Adaptive Vol Target ---
    trending_vol_target: float = Field(
        default=0.40,
        ge=0.05,
        le=1.0,
        description="Vol target in trending regime (aggressive).",
    )
    ranging_vol_target: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Vol target in ranging regime (moderate).",
    )
    volatile_vol_target: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Vol target in volatile regime (conservative).",
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
    def _validate_cross_fields(self) -> RegimeAdaptiveMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if not (self.fast_lookback < self.mid_lookback < self.slow_lookback):
            msg = (
                f"lookbacks must be ordered: fast ({self.fast_lookback}) "
                f"< mid ({self.mid_lookback}) < slow ({self.slow_lookback})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.slow_lookback, self.vol_window) + 10
