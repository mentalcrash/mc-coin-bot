"""Entropy-Gate Trend 4H 전략 설정.

Permutation Entropy(PermEn) 게이팅 + 3-scale Donchian breakout 앙상블.
Low entropy = predictable market = 진입 허용. High entropy = random = flat.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class EntropyGateTrend4hConfig(BaseModel):
    """Entropy-Gate Trend 4H 전략 설정.

    Permutation Entropy로 시장 예측 가능성을 판단하고,
    low-entropy 구간에서만 3-scale Donchian breakout을 활성화한다.

    Signal Logic:
        1. Rolling Permutation Entropy 계산 (embedding dim m, window)
        2. is_predictable = (perm_entropy < entropy_threshold)
        3. 3-scale Donchian breakout consensus 계산
        4. Gate: is_predictable이 False → signal = 0 (flat)
        5. strength = direction * |consensus| * vol_scalar * is_predictable

    Attributes:
        entropy_window: Entropy 계산 rolling window (4H bars).
        entropy_m: Embedding dimension (permutation 길이).
        entropy_delay: Permutation entropy delay parameter.
        entropy_threshold: 이 값 이상이면 random → flat.
        dc_scale_short: 단기 Donchian lookback (4H bars, ~15일).
        dc_scale_mid: 중기 Donchian lookback (4H bars, ~30일).
        dc_scale_long: 장기 Donchian lookback (4H bars, ~60일).
        entry_threshold: consensus 진입 임계값 (0~1).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 4H TF 연환산 계수 (365*6=2190).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Entropy Gate Parameters ---
    entropy_window: int = Field(
        default=48,
        ge=10,
        le=200,
        description="Entropy rolling window (4H bars, 48 = ~8일)",
    )
    entropy_m: int = Field(
        default=3,
        ge=2,
        le=7,
        description="Permutation embedding dimension",
    )
    entropy_delay: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Permutation entropy delay parameter",
    )
    entropy_threshold: float = Field(
        default=1.5,
        gt=0.0,
        le=5.0,
        description="이 값 이상이면 random → flat (PermEn 기준)",
    )

    # --- Donchian Multi-Scale Parameters ---
    dc_scale_short: int = Field(default=90, ge=10, le=300)
    dc_scale_mid: int = Field(default=180, ge=20, le=600)
    dc_scale_long: int = Field(default=360, ge=40, le=1200)
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
    annualization_factor: float = Field(default=2190.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> EntropyGateTrend4hConfig:
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
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.dc_scale_long, self.vol_window, self.entropy_window) + 10
