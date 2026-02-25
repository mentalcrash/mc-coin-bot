"""MVRV Cycle Trend 전략 설정.

MVRV Z-Score 사이클 레짐 필터 + 12H multi-lookback momentum.
BTC/ETH 전용 (CoinMetrics MVRV). SSRN 5225612 학술 검증.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class MvrvCycleTrendConfig(BaseModel):
    """MVRV Cycle Trend 전략 설정.

    Attributes:
        mvrv_zscore_window: MVRV Z-Score rolling window (bars, 1D 해상도).
        mvrv_bull_threshold: MVRV Z-Score 상승 사이클 진입 기준 (과소평가).
        mvrv_bear_threshold: MVRV Z-Score 하락 사이클 진입 기준 (과대평가).
        mom_fast: 빠른 모멘텀 lookback (12H bars).
        mom_slow: 느린 모멘텀 lookback (12H bars).
        mom_blend_weight: 빠른 모멘텀 가중치 (1-weight = 느린 가중치).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 12H TF 연환산 계수 (730.0).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- MVRV Cycle Parameters ---
    mvrv_zscore_window: int = Field(
        default=365, ge=90, le=730, description="MVRV Z-Score rolling window (1D bars)"
    )
    mvrv_bull_threshold: float = Field(
        default=-0.5, ge=-3.0, le=1.0, description="MVRV Z-Score bull regime entry"
    )
    mvrv_bear_threshold: float = Field(
        default=2.0, ge=0.5, le=7.0, description="MVRV Z-Score bear regime entry"
    )

    # --- Momentum Parameters (12H bars) ---
    mom_fast: int = Field(default=14, ge=3, le=60, description="Fast momentum lookback (12H bars)")
    mom_slow: int = Field(
        default=60, ge=20, le=180, description="Slow momentum lookback (12H bars)"
    )
    mom_blend_weight: float = Field(
        default=0.6, ge=0.1, le=0.9, description="Fast momentum blend weight"
    )

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=730.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> MvrvCycleTrendConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.mvrv_bull_threshold >= self.mvrv_bear_threshold:
            msg = (
                f"mvrv_bull_threshold ({self.mvrv_bull_threshold}) must be < "
                f"mvrv_bear_threshold ({self.mvrv_bear_threshold})"
            )
            raise ValueError(msg)
        if self.mom_fast >= self.mom_slow:
            msg = f"mom_fast ({self.mom_fast}) must be < mom_slow ({self.mom_slow})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.mom_slow, self.vol_window) + 10
