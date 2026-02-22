"""Stablecoin Velocity Regime 전략 설정.

스테이블코인 총 공급 velocity가 가속되면 시장 진입 자금 증가 → 가격 상승 선행.
실제 on-chain 데이터 없이 volume/market-cap proxy로 velocity 추정.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class StablecoinVelocityConfig(BaseModel):
    """Stablecoin Velocity Regime 전략 설정.

    Attributes:
        velocity_fast_window: Velocity proxy 빠른 평균 window.
        velocity_slow_window: Velocity proxy 느린 평균 window.
        zscore_window: Velocity z-score 계산 window.
        zscore_entry_threshold: Long 진입 z-score 임계값.
        zscore_exit_threshold: Short 진입 z-score 임계값 (음수).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 1D TF 연환산 계수 (365).
        atr_period: ATR 계산 기간.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    velocity_fast_window: int = Field(default=7, ge=3, le=30)
    velocity_slow_window: int = Field(default=30, ge=10, le=120)
    zscore_window: int = Field(default=60, ge=20, le=200)
    zscore_entry_threshold: float = Field(default=1.0, ge=0.3, le=3.0)
    zscore_exit_threshold: float = Field(default=-1.0, le=-0.3, ge=-3.0)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=365.0, gt=0.0)

    # --- Options ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> StablecoinVelocityConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.velocity_fast_window >= self.velocity_slow_window:
            msg = (
                f"velocity_fast_window ({self.velocity_fast_window}) must be < "
                f"velocity_slow_window ({self.velocity_slow_window})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            max(
                self.velocity_slow_window,
                self.zscore_window,
                self.vol_window,
                self.atr_period,
            )
            + 10
        )
