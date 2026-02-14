"""Funding Rate Carry (Vol-Conditioned) 전략 설정.

극단 FR에서 contrarian carry로 보험료 수취.
Vol 조건으로 저변동성 환경에서만 carry 포지션 유지 (high-vol whipsaw 방어).
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


class FRCarryVolConfig(BaseModel):
    """Funding Rate Carry (Vol-Conditioned) 전략 설정.

    Attributes:
        fr_lookback: Funding rate rolling 평균 윈도우 (4H bars).
        fr_zscore_window: Funding rate z-score 정규화 윈도우.
        fr_entry_threshold: 진입을 위한 최소 |avg_FR|.
        fr_extreme_zscore: 극단 FR z-score 기준 (이 이상이면 carry 진입).
        vol_condition_window: Vol 조건 계산 윈도우.
        vol_condition_pctile: Vol percentile 상한 (이 이하에서만 carry).
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 4H = 2190.0.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY drawdown 기준.
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄.
    """

    model_config = ConfigDict(frozen=True)

    # --- Funding Rate Parameters ---
    fr_lookback: int = Field(
        default=6,
        ge=1,
        le=30,
        description="FR rolling mean window (4H bars, 6 = 1 day).",
    )
    fr_zscore_window: int = Field(
        default=90,
        ge=10,
        le=500,
        description="FR z-score normalization window.",
    )
    fr_entry_threshold: float = Field(
        default=0.0001,
        ge=0.0,
        le=0.01,
        description="Minimum |avg_FR| for entry.",
    )
    fr_extreme_zscore: float = Field(
        default=1.5,
        ge=0.5,
        le=4.0,
        description="Extreme FR z-score threshold for carry entry.",
    )

    # --- Vol Conditioning ---
    vol_condition_window: int = Field(
        default=60,
        ge=10,
        le=200,
        description="Vol condition percentile rank window.",
    )
    vol_condition_pctile: float = Field(
        default=0.7,
        ge=0.3,
        le=1.0,
        description="Max vol percentile for carry (below = low-vol, carry OK).",
    )

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=2190.0, gt=0)

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
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.fr_zscore_window, self.vol_condition_window, self.vol_window) + 10
