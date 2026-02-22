"""Kurtosis Carry 전략 설정.

고첨도(fat tail) → 리스크 프리미엄 확대 → 저첨도 전환 시 프리미엄 수취.
Amaya et al. (JFE 2015) 학술 근거.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class KurtosisCarryConfig(BaseModel):
    """Kurtosis Carry 전략 설정.

    Attributes:
        kurtosis_window: Rolling kurtosis 계산 window (bars).
        kurtosis_long_window: 장기 kurtosis 기준선 window.
        zscore_window: Kurtosis z-score 계산 window.
        high_kurtosis_zscore: 고첨도 진입 z-score 임계값.
        low_kurtosis_zscore: 저첨도 short 진입 z-score 임계값.
        momentum_lookback: 모멘텀 방향 확인 lookback.
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
    kurtosis_window: int = Field(default=30, ge=10, le=120)
    kurtosis_long_window: int = Field(default=90, ge=30, le=365)
    zscore_window: int = Field(default=60, ge=20, le=200)
    high_kurtosis_zscore: float = Field(default=1.0, ge=0.3, le=3.0)
    low_kurtosis_zscore: float = Field(default=-0.5, le=-0.1, ge=-3.0)
    momentum_lookback: int = Field(default=20, ge=5, le=60)

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
    def _validate_cross_fields(self) -> KurtosisCarryConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.kurtosis_window >= self.kurtosis_long_window:
            msg = (
                f"kurtosis_window ({self.kurtosis_window}) must be < "
                f"kurtosis_long_window ({self.kurtosis_long_window})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            max(
                self.kurtosis_long_window,
                self.zscore_window,
                self.vol_window,
                self.atr_period,
                self.momentum_lookback,
            )
            + 10
        )
