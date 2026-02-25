"""Multi-Source Directional Composite 전략 설정.

3개 직교 데이터 소스(OHLCV momentum + stablecoin flow proxy + F&G sentiment)의
majority vote 기반 방향 결정. 약한 알파의 포트폴리오 결합 이론.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class MultiSourceCompositeConfig(BaseModel):
    """Multi-Source Directional Composite 전략 설정.

    3개 직교 소스의 directional sub-signal을 majority vote로 결합.
    최소 2/3 합의 시 진입, 전원 합의 시 conviction 상승.

    Attributes:
        mom_fast: 빠른 EMA 기간 (추세 방향 감지).
        mom_slow: 느린 EMA 기간 (추세 방향 감지).
        mom_lookback: rolling return lookback (모멘텀 확인용).
        velocity_fast_window: stablecoin velocity proxy fast MA window.
        velocity_slow_window: stablecoin velocity proxy slow MA window.
        fg_delta_window: F&G 변화율 계산 window.
        fg_smooth_window: F&G delta smoothing window.
        fg_threshold: F&G delta 방향 판별 임계값.
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 1D TF 연환산 계수 (365).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Momentum Parameters (Source 1: OHLCV) ---
    mom_fast: int = Field(default=10, ge=3, le=50)
    mom_slow: int = Field(default=30, ge=10, le=120)
    mom_lookback: int = Field(default=20, ge=5, le=120)

    # --- Stablecoin Velocity Proxy Parameters (Source 2) ---
    velocity_fast_window: int = Field(default=7, ge=3, le=30)
    velocity_slow_window: int = Field(default=30, ge=10, le=120)

    # --- Fear & Greed Sentiment Parameters (Source 3) ---
    fg_delta_window: int = Field(default=7, ge=3, le=30)
    fg_smooth_window: int = Field(default=5, ge=3, le=20)
    fg_threshold: float = Field(default=2.0, ge=0.0, le=20.0)

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
    def _validate_cross_fields(self) -> MultiSourceCompositeConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.mom_fast >= self.mom_slow:
            msg = f"mom_fast ({self.mom_fast}) must be < mom_slow ({self.mom_slow})"
            raise ValueError(msg)
        if self.velocity_fast_window >= self.velocity_slow_window:
            msg = (
                f"velocity_fast_window ({self.velocity_fast_window}) "
                f"must be < velocity_slow_window ({self.velocity_slow_window})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            max(
                self.mom_slow,
                self.mom_lookback,
                self.velocity_slow_window,
                self.fg_delta_window + self.fg_smooth_window,
                self.vol_window,
            )
            + 10
        )
