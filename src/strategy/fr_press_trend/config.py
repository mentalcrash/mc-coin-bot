"""Funding Pressure Trend 전략 설정.

가격 추세(SMA cross)가 primary alpha,
FR z-score는 포지셔닝 압력/군중 밀집도 리스크 필터로만 사용.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class FrPressTrendConfig(BaseModel):
    """Funding Pressure Trend 전략 설정.

    Attributes:
        sma_fast: 빠른 SMA 기간 (~40시간 at 4H).
        sma_slow: 느린 SMA 기간 (~7일 at 4H).
        er_window: ER 추세 품질 필터 기간.
        er_threshold: 최소 ER (이상이면 trending).
        fr_ma_window: FR 이동평균 윈도우.
        fr_zscore_window: FR z-score 정규화 윈도우.
        fr_aligned_threshold: FR 정렬 판단 임계 (이하면 aligned).
        fr_extreme_threshold: FR 극단 판단 임계 (이상이면 block).
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 4H TF 연환산 계수.
        atr_period: ATR 계산 기간.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY drawdown 기준.
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄.
    """

    model_config = ConfigDict(frozen=True)

    # --- Trend Parameters ---
    sma_fast: int = Field(default=10, ge=5, le=30)
    sma_slow: int = Field(default=42, ge=20, le=120)
    er_window: int = Field(default=20, ge=10, le=60)
    er_threshold: float = Field(default=0.20, ge=0.05, le=0.5)

    # --- Funding Rate Filter ---
    fr_ma_window: int = Field(default=21, ge=3, le=60)
    fr_zscore_window: int = Field(default=42, ge=10, le=120)
    fr_aligned_threshold: float = Field(default=1.5, ge=0.5, le=3.0)
    fr_extreme_threshold: float = Field(default=2.5, ge=1.5, le=5.0)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=2190.0, gt=0.0)

    # --- Options ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> Self:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.sma_fast >= self.sma_slow:
            msg = f"sma_fast ({self.sma_fast}) must be < sma_slow ({self.sma_slow})"
            raise ValueError(msg)
        if self.fr_aligned_threshold >= self.fr_extreme_threshold:
            msg = (
                f"fr_aligned_threshold ({self.fr_aligned_threshold}) "
                f"must be < fr_extreme_threshold ({self.fr_extreme_threshold})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.sma_slow, self.fr_zscore_window, self.vol_window) + 10
