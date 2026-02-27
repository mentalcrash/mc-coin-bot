"""DVOL-Trend 8H 전략 설정.

Deribit DVOL(내재변동성) percentile 기반 position sizing + 8H multi-scale Donchian channel.
BTC/ETH only.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class DvolTrend8hConfig(BaseModel):
    """DVOL-Trend 8H 전략 설정.

    Deribit DVOL percentile로 IV regime을 판단하여 position sizing을 조절하고,
    3-scale Donchian Channel breakout consensus로 추세 방향을 결정한다.

    DVOL Regime Logic:
        1. DVOL percentile = rolling rank (dvol_percentile_window)
        2. percentile < dvol_low_threshold  → size *= dvol_low_multiplier  (저IV: 확대)
        3. percentile > dvol_high_threshold → size *= dvol_high_multiplier (고IV: 축소)
        4. else → size *= 1.0 (중립)

    Signal Logic:
        1. 각 scale(22/66/132) Donchian Channel breakout 시그널
        2. consensus = mean(signal_short, signal_mid, signal_long)
        3. direction = sign(consensus) if |consensus| >= entry_threshold else 0
        4. strength = |consensus| * vol_scalar * dvol_size_mult

    Attributes:
        dvol_percentile_window: DVOL percentile rolling window (bars).
        dvol_low_threshold: 저IV regime 기준 (0~1).
        dvol_high_threshold: 고IV regime 기준 (0~1).
        dvol_low_multiplier: 저IV regime size 배율.
        dvol_high_multiplier: 고IV regime size 배율.
        dc_scale_short: 단기 Donchian lookback (bars).
        dc_scale_mid: 중기 Donchian lookback (bars).
        dc_scale_long: 장기 Donchian lookback (bars).
        entry_threshold: consensus 진입 임계값 (0~1).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 8H TF 연환산 계수 (1095).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- DVOL Parameters ---
    dvol_percentile_window: int = Field(default=252, ge=20, le=1000)
    dvol_low_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="이 percentile 이하 → 저IV regime (size 확대)",
    )
    dvol_high_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="이 percentile 이상 → 고IV regime (size 축소)",
    )
    dvol_low_multiplier: float = Field(
        default=1.2,
        gt=0.0,
        le=3.0,
        description="저IV regime size 배율",
    )
    dvol_high_multiplier: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="고IV regime size 배율",
    )

    # --- Donchian Channel Parameters ---
    dc_scale_short: int = Field(default=22, ge=5, le=100)
    dc_scale_mid: int = Field(default=66, ge=10, le=300)
    dc_scale_long: int = Field(default=132, ge=20, le=600)
    entry_threshold: float = Field(
        default=0.34,
        ge=0.0,
        le=1.0,
        description="consensus 진입 임계값. 0.34 = 3개 중 2개 합의 필요",
    )

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=1095.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> DvolTrend8hConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if not (self.dc_scale_short < self.dc_scale_mid < self.dc_scale_long):
            msg = (
                f"dc_scale_short ({self.dc_scale_short}) < dc_scale_mid ({self.dc_scale_mid}) "
                f"< dc_scale_long ({self.dc_scale_long}) 필수"
            )
            raise ValueError(msg)
        if self.dvol_low_threshold >= self.dvol_high_threshold:
            msg = (
                f"dvol_low_threshold ({self.dvol_low_threshold}) must be < "
                f"dvol_high_threshold ({self.dvol_high_threshold})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.dc_scale_long, self.vol_window, self.dvol_percentile_window) + 10
