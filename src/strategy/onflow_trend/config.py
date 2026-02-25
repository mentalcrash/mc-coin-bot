"""OnFlow Trend 전략 설정.

거래소 순입출금 + MVRV z-score 기반 추세추종.
8H OHLCV primary + 1D On-chain context. BTC/ETH 전용.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class OnflowTrendConfig(BaseModel):
    """OnFlow Trend 전략 설정.

    Attributes:
        flow_zscore_window: 거래소 순입출금 z-score rolling window.
        flow_long_z: 순유출 롱 진입 z-score 임계값 (< -threshold → 축적 → Long).
        flow_exit_z: 순유입 청산 z-score 임계값 (> threshold → 분배 → 방어).
        mvrv_undervalued: MVRV 저평가 임계값 (< → 확신도 증가).
        mvrv_overheated: MVRV 과열 임계값 (> → 확신도 감소/방어).
        trend_ema_fast: 가격 추세 판단 빠른 EMA (8H bars).
        trend_ema_slow: 가격 추세 판단 느린 EMA (8H bars).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 8H TF 연환산 계수 (1095).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    flow_zscore_window: int = Field(default=90, ge=20, le=365)
    flow_long_z: float = Field(default=-1.0, ge=-3.0, le=0.0)
    flow_exit_z: float = Field(default=1.0, ge=0.0, le=3.0)
    mvrv_undervalued: float = Field(default=1.0, ge=0.3, le=2.0)
    mvrv_overheated: float = Field(default=3.5, ge=2.0, le=6.0)
    trend_ema_fast: int = Field(default=12, ge=3, le=50)
    trend_ema_slow: int = Field(default=36, ge=10, le=200)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.30, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=1095.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> OnflowTrendConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.mvrv_undervalued >= self.mvrv_overheated:
            msg = (
                f"mvrv_undervalued ({self.mvrv_undervalued}) must be < "
                f"mvrv_overheated ({self.mvrv_overheated})"
            )
            raise ValueError(msg)
        if self.trend_ema_fast >= self.trend_ema_slow:
            msg = (
                f"trend_ema_fast ({self.trend_ema_fast}) must be < "
                f"trend_ema_slow ({self.trend_ema_slow})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.flow_zscore_window, self.trend_ema_slow, self.vol_window) + 10
