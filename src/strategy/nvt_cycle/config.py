"""NVT Cycle Signal 전략 설정.

NVT(Network Value to Transactions) = 크립토의 P/E 비율.
네트워크 사용량 대비 시가총액 과대/과소평가 측정. BTC/ETH 전용.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class NvtCycleConfig(BaseModel):
    """NVT Cycle Signal 전략 설정.

    Attributes:
        nvt_window: NVT ratio smoothing 기간.
        nvt_zscore_window: NVT z-score 계산 기간.
        overbought_threshold: 과대평가 z-score 임계값.
        oversold_threshold: 과소평가 z-score 임계값 (음수).
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 1D TF 연환산 계수.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준.
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    nvt_window: int = Field(default=14, ge=3, le=60)
    nvt_zscore_window: int = Field(default=90, ge=30, le=365)
    overbought_threshold: float = Field(default=1.0, ge=0.0, le=3.0)
    oversold_threshold: float = Field(default=-1.0, ge=-3.0, le=0.0)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=365.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> NvtCycleConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.nvt_window + self.nvt_zscore_window, self.vol_window) + 10
