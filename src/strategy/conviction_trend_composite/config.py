"""Conviction Trend Composite 전략 설정.

가격 모멘텀 PRIMARY + OBV 거래량 구조/RV 변동성 비율을 conviction modifier로 활용.
독립 데이터 소스의 합의가 높을 때만 진입. 레짐 확률 가중 사이징.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class ConvictionTrendCompositeConfig(BaseModel):
    """Conviction Trend Composite 전략 설정.

    Price momentum이 방향, OBV 거래량 구조 + RV ratio가 conviction.
    factor 역할 분리: 방향 vs 강도로 과적합 완화.

    Attributes:
        mom_lookback: 가격 모멘텀 rolling return window (1D bars).
        mom_fast: 빠른 EMA 기간 (추세 방향 감지).
        mom_slow: 느린 EMA 기간 (추세 방향 감지).
        obv_fast: OBV 빠른 EMA 기간.
        obv_slow: OBV 느린 EMA 기간.
        rv_short_window: 단기 RV 계산 window.
        rv_long_window: 장기 RV 계산 window.
        conviction_threshold: 최소 conviction 점수 (0~1, 이상에서만 진입).
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

    # --- OBV Parameters ---
    obv_fast: int = Field(
        default=10,
        ge=3,
        le=50,
        description="OBV fast EMA period.",
    )
    obv_slow: int = Field(
        default=30,
        ge=10,
        le=120,
        description="OBV slow EMA period.",
    )

    # --- Realized Volatility Ratio Parameters ---
    rv_short_window: int = Field(
        default=10,
        ge=3,
        le=30,
        description="Short-term RV calculation window.",
    )
    rv_long_window: int = Field(
        default=60,
        ge=20,
        le=252,
        description="Long-term RV calculation window.",
    )

    # --- Conviction Threshold ---
    conviction_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum conviction score to enter (0~1).",
    )

    # --- Regime-Adaptive Vol Target ---
    trending_vol_target: float = Field(
        default=0.40,
        ge=0.05,
        le=1.0,
        description="Vol target in trending regime (aggressive).",
    )
    ranging_vol_target: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Vol target in ranging regime (conservative).",
    )
    volatile_vol_target: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Vol target in volatile regime (ultra-conservative).",
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
    def _validate_cross_fields(self) -> ConvictionTrendCompositeConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.mom_fast >= self.mom_slow:
            msg = f"mom_fast ({self.mom_fast}) must be < mom_slow ({self.mom_slow})"
            raise ValueError(msg)
        if self.obv_fast >= self.obv_slow:
            msg = f"obv_fast ({self.obv_fast}) must be < obv_slow ({self.obv_slow})"
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
                self.mom_lookback,
                self.mom_slow,
                self.obv_slow,
                self.rv_long_window,
                self.vol_window,
            )
            + 10
        )
