"""Carry-Momentum Convergence 전략 설정.

가격 모멘텀 PRIMARY + FR z-score를 conviction modifier로 활용.
FR 캐리 수익이 아닌 추세 건강도 측정. 가격-FR 수렴 시 강한 추세, 발산 시 피로.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class CarryMomConvergenceConfig(BaseModel):
    """Carry-Momentum Convergence 전략 설정.

    Price momentum이 alpha source, FR z-score는 conviction modifier.
    가격 추세와 FR 방향이 수렴하면 높은 conviction, 발산하면 낮은 conviction.

    Attributes:
        mom_lookback: 가격 모멘텀 rolling return window (1D bars).
        mom_fast: 빠른 EMA 기간 (추세 방향 감지).
        mom_slow: 느린 EMA 기간 (추세 방향 감지).
        fr_lookback: Funding rate rolling mean window (1D bars).
        fr_zscore_window: FR z-score normalization window.
        convergence_boost: FR-price 수렴 시 conviction 승수.
        divergence_penalty: FR-price 발산 시 conviction 감쇄 계수.
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 1D TF 연환산 계수 (365).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Momentum Parameters ---
    mom_lookback: int = Field(
        default=20,
        ge=5,
        le=120,
        description="Price momentum rolling return window (1D bars).",
    )
    mom_fast: int = Field(
        default=10,
        ge=3,
        le=50,
        description="Fast EMA period for trend direction.",
    )
    mom_slow: int = Field(
        default=30,
        ge=10,
        le=120,
        description="Slow EMA period for trend direction.",
    )

    # --- Funding Rate Parameters ---
    fr_lookback: int = Field(
        default=3,
        ge=1,
        le=30,
        description="FR rolling mean window (1D bars).",
    )
    fr_zscore_window: int = Field(
        default=90,
        ge=10,
        le=365,
        description="FR z-score normalization window.",
    )

    # --- Convergence Parameters ---
    convergence_boost: float = Field(
        default=1.5,
        ge=1.0,
        le=3.0,
        description="Conviction multiplier when FR confirms price trend.",
    )
    divergence_penalty: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Conviction damping when FR diverges from price trend.",
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
    def _validate_cross_fields(self) -> CarryMomConvergenceConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.mom_fast >= self.mom_slow:
            msg = f"mom_fast ({self.mom_fast}) must be < mom_slow ({self.mom_slow})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            max(
                self.mom_lookback,
                self.mom_slow,
                self.fr_zscore_window,
                self.vol_window,
            )
            + 10
        )
